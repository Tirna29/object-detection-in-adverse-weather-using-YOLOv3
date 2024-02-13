import numpy as np
import os
import cv2
import math
import config.yolov3_config_voc as cfg
import random

# only use the image including the labeled instance objects for training
def load_annotations(annot_path):
    print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations


# print('*****************Add haze offline***************************')
def parse_annotation(annotation):

    line = annotation.split()
    image_path = line[0]


    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = cv2.imread(image_path)

    def AddHaz_loop(img_f, center, size, beta, A):
        (row, col, chs) = img_f.shape

        for j in range(row):
            for l in range(col):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-beta * d)
                img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
        return img_f

    img_f = image / 255
    (row, col, chs) = image.shape
    A = 0.5
    # beta = 0.08
    beta = random.randint(0, 9)
    beta = 0.01 * beta + 0.05
    size = math.sqrt(max(row, col))
    center = (row // 2, col // 2)
    foggy_image = AddHaz_loop(img_f, center, size, beta, A)
    img_f = np.clip(foggy_image * 255, 0, 255)
    img_f = img_f.astype(np.uint8)
    new_image_path = os.path.join("./data/", 'FoggyImages', image_path.split('/')[-1])
    cv2.imwrite(new_image_path, img_f)




if __name__ == '__main__':
    an = load_annotations('./data/data_fog/train_annotation.txt')

    ll = len(an)

    for j in range(ll):
        parse_annotation(an[j])
        print(an[j])

    # an = load_annotations('./data/data_fog/test_annotation.txt')
    #
    # ll = len(an)
    # print(ll)
    # for j in range(ll):
    #     parse_annotation(an[j])