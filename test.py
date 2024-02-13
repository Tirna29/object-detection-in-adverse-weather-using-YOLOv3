import math
from torch import nn
from easydict import EasyDict as edict
import utils.gpu as gpu
from YOLOV3.model.cnn_pp import Dip_Cnn
from filters import DefogFilter, ImprovedWhiteBalanceFilter, GammaFilter, ToneFilter, ContrastFilter, UsmFilter
from model.yolov3 import Yolov3

from utils.tools import *
from eval.evaluator import Evaluator
import argparse
import config.yolov3_config_voc as cfg
from utils.visualize import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

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


def Dip_filters(features, img, cfg=dip_cfg):
    filtered_image_batch = img
    B, C, W, H = img.shape
    dark = torch.zeros([B, W, H], dtype=torch.float32).to(img.device)
    defog_A = torch.zeros([B, C], dtype=torch.float32).to(img.device)
    IcA = torch.zeros([B, W, H], dtype=torch.float32).to(img.device)
    for i in range(B):
        dark_i = DarkChannel(img[i])
        defog_A_i = AtmLight(img[i], dark_i)
        IcA_i = DarkIcA(img[i], defog_A_i)
        dark[i, ...] = dark_i
        defog_A[i, ...] = defog_A_i
        IcA[i, ...] = IcA_i
    IcA = IcA.unsqueeze(-1)

    filters = cfg.filters
    filters = [x(filtered_image_batch, cfg) for x in filters]
    filter_features = features
    for j, filter in enumerate(filters):
        filtered_image_batch, filter_parameter = filter.apply(
            filtered_image_batch, filter_features, defog_A, IcA)

    return filtered_image_batch


def DarkChannel(im):
    R = im[0, :, :]
    G = im[1, :, :]
    B = im[2, :, :]
    dc = torch.min(torch.min(R, G), B)
    return dc


def AtmLight(im, dark):
    c, h, w = im.shape
    imsz = h * w
    numpx = int(max(torch.floor(torch.tensor(imsz) / 1000), torch.tensor(1)))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(3, imsz)

    indices = torch.argsort(darkvec)
    indices = indices[(imsz - numpx):imsz]

    atmsum = torch.zeros([3, 1]).to(imvec.device)
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[:, indices[ind]]

    A = atmsum / numpx
    return A.reshape(1, 3)


def DarkIcA(im, A):
    c, h, w = im.shape
    im3 = torch.zeros([c, h, w]).to(im.device)
    for ind in range(0, 3):
        im3[ind, :, :] = im[ind, :, :] / A[0, ind]
    return DarkChannel(im3)

class Tester(object):
    def __init__(self,
                 weight_path=None,
                 gpu_id=0,
                 img_size=544,
                 visiual=None,
                 eval=False
                 ):
        self.img_size = img_size
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]

        self.__visiual = visiual
        self.__eval = eval
        self.__classes = cfg.DATA["CLASSES"]

        self.__model = Yolov3().to(self.__device)

        self.__load_model_weights(weight_path)
        self.cnn = Dip_Cnn().to(self.__device)
        self.cnn.load_state_dict(torch.load('weight/best_cnn.pt'))
        self.cnn.eval()
        self.__evalter = Evaluator(self.__model, visiual=True)



    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt

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



    def test(self):
        if self.__visiual:
            imgs = os.listdir(self.__visiual)

            for v in imgs:
                path = os.path.join(self.__visiual, v)
                print("test images : {}".format(path))

                img = cv2.imread(path)
                assert img is not None


                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0).float().to(self.__device)

                img = img / 255
                resize_images = nn.functional.interpolate(img.clone(), size=(256, 256), mode='bilinear', align_corners=False)

                # Pass the image through the CNN
                features = self.cnn(resize_images)

                # Apply the filters
                img = Dip_filters(features, img)

                img = img * 255



                #Convert image to numpy
                img = img.cpu().detach().numpy()
                img = img.squeeze(0)
                img = img.transpose(1, 2, 0)




                bboxes_prd = self.__evalter.get_bbox(img, multi_test=False, flip_test=False)

                if bboxes_prd.shape[0] != 0:
                        boxes = bboxes_prd[..., :4]
                        class_inds = bboxes_prd[..., 5].astype(np.int32)
                        scores = bboxes_prd[..., 4]

                        visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
                        path = os.path.join(cfg.PROJECT_PATH, "data/results/{}".format(v))

                        cv2.imwrite(path, img)
                        print("saved images : {}".format(path))


if __name__ == "__main__":
    os.chdir('/tmp/pycharm_project_780/YOLOV3')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/best_yolo.pt', help='weight file path')
    parser.add_argument('--visiual', type=str, default='./data/FoggyImages', help='test data path or None')
    parser.add_argument('--eval', action='store_true', default=True, help='eval the mAP or not')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester( weight_path=opt.weight_path,
            gpu_id=opt.gpu_id,
            eval=opt.eval,
            visiual=opt.visiual).test()
