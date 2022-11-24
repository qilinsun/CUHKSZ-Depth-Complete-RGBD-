# Import modules
import cv2
import numpy as np
import math
import sys, time
import os


# path = "EMDC-PyTorch/Submit/validation_input/data0929/aligned_depth"
path = "new_aligned_depth"
img_names = os.listdir(path)

def fill_max(img):
    H, W = img.shape[0], img.shape[1]
    dst = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            if img[i,j] == 0:
                if (i-1<0 or i+1>=H or j-1<0 or j+1>=W):
                    continue
                dst[i,j] = np.max([img[i-1,j-1],img[i-1,j],img[i-1,j+1],
                             img[i,j-1],img[i,j],img[i,j+1],
                             img[i+1,j-1],img[i+1,j],img[i+1,j+1]])
            else:
                dst[i,j] = img[i,j]
    return dst            


for name in img_names:
    start_time = time.time()
    img_path = os.path.join(path, name)

    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)

    print("processing", name)
    filled_img = fill_max(img)
    filled_img = filled_img.astype(np.uint16)
    dep_path = os.path.join("new_max_aligned_depth", name)
    cv2.imwrite(dep_path, filled_img)
    end_time = time.time()
    print("it takes", end_time - start_time)
