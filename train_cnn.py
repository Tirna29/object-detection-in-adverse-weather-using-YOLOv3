import math
from easydict import EasyDict as edict


import utils.gpu as gpu

from YOLOV3.model.cnn_pp import Dip_Cnn
from YOLOV3.test import Dip_filters
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
from filters import DefogFilter, ImprovedWhiteBalanceFilter, GammaFilter, ToneFilter, ContrastFilter, UsmFilter
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import DataLoader
import utils.datasets as data

import argparse
from utils.tools import *

import config.yolov3_config_voc as cfg
from utils import cosine_lr_scheduler


import os
os.environ["CUDA_VISIBLE_DEVICES"]='6'

dip_cfg = edict()
dip_cfg.filters = [
DefogFilter, ImprovedWhiteBalanceFilter, GammaFilter,
    ToneFilter, ContrastFilter, UsmFilter
]


dip_cfg.num_filter_parameters = 15

dip_cfg.defog_begin_param = 0

dip_cfg.wb_begin_param = 1
dip_cfg.gamma_begin_param = 4
dip_cfg.tone_begin_param = 5
dip_cfg.contrast_begin_param = 13
dip_cfg.usm_begin_param = 14

dip_cfg.curve_steps = 8
dip_cfg.gamma_range = 3
dip_cfg.exposure_range = 3.5
dip_cfg.wb_range = 1.1
dip_cfg.color_curve_range = (0.90, 1.10)
dip_cfg.lab_curve_range = (0.90, 1.10)
dip_cfg.tone_curve_range = (0.5, 2)
dip_cfg.defog_range = (0.1, 1.0)
dip_cfg.usm_range = (0.0, 5)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)

# Code adapted from https://github.com/noefford/Image-Adaptive-YOLO-pytorch
class CNN_Trainer(object):
    def __init__(self, weight_path, resume, gpu_id):
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True)

        self.yolov3 = Yolov3().to(self.device)

        self.cnn = Dip_Cnn().to(self.device)

        self.optimizer = optim.Adam(self.cnn.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                    weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        self.optimizer_yolo = optim.Adam(self.yolov3.parameters(), lr=cfg.TRAIN["LR_INIT"]
                                    ,weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.__load_model_weights(weight_path, resume)
        self.__load_cnn_model_weights(weight_path, resume)
        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))

    def fog_image(self, image):
        def AddHaz_vectorized(img_f, center, size, beta, A):
            row, col, chs = img_f.shape
            x, y = np.meshgrid(np.arange(col), np.arange(row))
            d = -0.04 * np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) + size
            td = np.exp(-beta * d)
            img_f = img_f * td[..., np.newaxis] + A * (1 - td[..., np.newaxis])
            return img_f

        img_f = image.astype(np.float32) / 255
        row, col, chs = image.shape
        A = 0.5
        beta = 0.01 * random.randint(0, 9) + 0.05
        size = math.sqrt(max(row, col))
        center = (row // 2, col // 2)
        foggy_image = AddHaz_vectorized(img_f, center, size, beta, A)
        img_f = np.clip(foggy_image * 255, 0, 255).astype(np.uint8)

        return img_f
    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(os.path.split(weight_path)[0], "last_yolo.pt")
            chkpt = torch.load(last_weight, map_location=self.device)
            self.yolov3.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            self.yolov3.load_darknet_weights(weight_path)

    def __load_cnn_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(os.path.split(weight_path)[0], "last_cnn.pt")
            chkpt = torch.load(last_weight, map_location=self.device)
            self.cnn.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            self.cnn.apply(weights_init)
    def __save_yolo_model_weights(self, epoch,mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best_yolo_cnn.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last_yolo.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt' % epoch))
        del chkpt

    def __save_cnn_model_weights(self, epoch,mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best_cnn.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last_cnn.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.cnn.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt' % epoch))
        del chkpt
    def train(self):
        print("Train datasets number is : {}".format(len(self.train_dataset)))


        for epoch in range(self.start_epoch, self.epochs):
            self.cnn.train()
            self.yolov3.train()
            mloss = torch.zeros(4)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(self.train_dataloader):


                for j in range(imgs.shape[0]):

                    img = imgs[j].permute(1,2,0).to("cpu").detach().numpy()

                    img = self.fog_image(img)
                    img = torch.tensor(img).permute(2,0,1)
                    imgs[j] = img

                self.scheduler.step(len(self.train_dataloader)*epoch + i)

                imgs = imgs.to(self.device)


                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                imgs = imgs / 255


                resize_images = nn.functional.interpolate(imgs.clone(), size=(256, 256), mode='bilinear', align_corners=False)



                features = self.cnn(resize_images)


                imgs = Dip_filters(features, imgs)

                imgs = imgs * 255


                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)


                self.optimizer.zero_grad()
                self.optimizer_yolo.zero_grad()
                loss.backward()

                self.optimizer.step()
                self.optimizer_yolo.step()
                
                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i%10==0:
                    s = ('Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                         'lr: %g') % (epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, mloss[0],mloss[1], mloss[2], mloss[3],
                                      self.optimizer.param_groups[0]['lr'])
                    print(s)

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(9,20)) * 32
                    print("multi_scale_img_size : {}".format(self.train_dataset.img_size))

            mAP = 0

            self.__save_yolo_model_weights(epoch,mAP)
            self.__save_cnn_model_weights(epoch,mAP)

            print('best mAP : %g' % (self.best_mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    CNN_Trainer(weight_path=opt.weight_path,
            resume=opt.resume,
            gpu_id=opt.gpu_id).train()