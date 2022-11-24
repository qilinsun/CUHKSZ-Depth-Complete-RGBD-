import cv2
import os
import numpy as np

PATH = "aligned_depth"

origin_rgb_imgs = os.listdir(PATH)

for img_name in origin_rgb_imgs:
    origin_img_path = os.path.join(PATH, img_name)
    sparse_img = np.zeros((1080,1920))
    sample_idx = np.random.choice(range(1920),480)
    origin_img = cv2.imread(origin_img_path, cv2.IMREAD_ANYDEPTH)
    # print(origin_img.shape)
    # print(sparse_img.shape)
    for i in range(1080):
        sample_idx = np.random.choice(range(1920),20)
        sparse_img[i, sample_idx] = origin_img[i, sample_idx]
    sparse_img = sparse_img.astype(np.uint16)
    cv2.imwrite(os.path.join("sparse_depth", img_name), sparse_img)