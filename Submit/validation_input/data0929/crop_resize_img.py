from ctypes import resize
import cv2
import os
import numpy as np

PATH = "origin_rgb"

origin_rgb_imgs = os.listdir(PATH)

for img_name in origin_rgb_imgs:
    origin_rgb_img_path = os.path.join(PATH, img_name)
    origin_bgr_img = cv2.imread(origin_rgb_img_path)
    crop_img = origin_bgr_img[:, 240:1680] #1920*1080 -> 1440*1080
    print(crop_img.shape)
    resize_img = cv2.resize(crop_img, (640, 480), interpolation=cv2.INTER_AREA) # 1920*1080 640*480
    cv2.imwrite(os.path.join("crop_resize_img", img_name), resize_img)

