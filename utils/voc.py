import sys
sys.path.append("..")
import xml.etree.ElementTree as ET
import config.yolov3_config_voc as cfg
import os
from tqdm import tqdm



def parse_voc_annotation(data_path, file_type, anno_path, use_difficult_bbox=False):
    classes = cfg.DATA["CLASSES"]
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', file_type+'.txt')
    with open(img_inds_file, 'r') as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open(anno_path, 'a') as f:
        for image_id in tqdm(image_ids):
            if file_type == 'foggy':
                image_path = os.path.join(cfg.DATA_PATH, 'FoggyImages', image_id + '.jpg')
            else:
                image_path = os.path.join(data_path, 'JPEGImages', image_id + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_id + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                if obj.find('name').text.lower().strip() not in classes:
                    continue
                bbox = obj.find('bndbox')
                class_id = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                ymin = bbox.find('ymin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
            annotation += '\n'
            # print(annotation)
            if len(annotation.strip().split(' ')) <= 1:
                continue
            f.write(annotation)
    return len(image_ids)


if __name__ =="__main__":
    # train_set :  VOC2007_trainval 和 VOC2012_trainval
    train_data_path_2007 = os.path.join(cfg.DATA_PATH, 'VOCtrainval-2007', 'VOCdevkit', 'VOC2007')
    # train_data_path_2012 = os.path.join(cfg.DATA_PATH, 'VOCtrainval-2012', 'VOCdevkit', 'VOC2012')
    train_annotation_path = os.path.join('../data/data_fog', 'train_annotation.txt')
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)

    # val_set   : VOC2007_test
    test_data_path_2007 = os.path.join(cfg.DATA_PATH, 'VOCtest-2007', 'VOCdevkit', 'VOC2007')
    test_annotation_path = os.path.join('../data/data_fog', 'test_annotation.txt')
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)

    foggy_data_path_2007 = os.path.join(cfg.DATA_PATH, 'VOCtrainval-2007', 'VOCdevkit', 'VOC2007')
    foggy_annotation_path = os.path.join('../data/data_fog', 'foggy_annotation.txt')
    if os.path.exists(foggy_annotation_path):
        os.remove(foggy_annotation_path)

    len_train = parse_voc_annotation(train_data_path_2007, "trainval", train_annotation_path, use_difficult_bbox=False)
            # parse_voc_annotation(train_data_path_2012, "trainval", train_annotation_path, use_difficult_bbox=False)
    len_test = parse_voc_annotation(test_data_path_2007, "test", test_annotation_path, use_difficult_bbox=False)
    len_foggy = parse_voc_annotation(foggy_data_path_2007, "foggy", foggy_annotation_path, use_difficult_bbox=False)
    print("The number of images for train,test and foggy  are :train : {0} | test : {1} | foggy: {2}".format(len_train, len_test, len_foggy))
