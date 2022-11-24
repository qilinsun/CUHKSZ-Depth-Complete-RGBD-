import cv2
import os
import numpy as np

PATH_rgb = "origin_rgb"
PATH_dep = "sparse_depth"
origin_rgb_imgs = os.listdir(PATH_rgb)
origin_dep_imgs = os.listdir(PATH_dep)
for img_name in origin_rgb_imgs:
    origin_img_path = os.path.join(PATH_rgb, img_name)
    origin_img = cv2.imread(origin_img_path)
    crop_img = origin_img[270:270+540,480:480+960]
    cv2.imwrite(os.path.join("crop_rgb", img_name), crop_img)

for img_name in origin_dep_imgs:
    origin_img_path = os.path.join(PATH_dep, img_name)
    origin_img = cv2.imread(origin_img_path, cv2.IMREAD_ANYDEPTH)
    crop_img = origin_img[270:270+540,480:480+960]
    crop_img = crop_img.astype(np.uint16)
    cv2.imwrite(os.path.join("crop_depth", img_name), crop_img)
