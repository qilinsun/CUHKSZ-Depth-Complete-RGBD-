import os
import cv2
import numpy as np
import os

PATH = "data0929/result/eval/0/"


pred_dep = cv2.imread("gallary/eval_gallary/gallary_gradient/eval/0/pred_dep_frame-000000_0.png", cv2.IMREAD_ANYDEPTH)    
    
raw_dep = cv2.imread("gallary/eval_gallary/gallary_gradient/eval/0/raw_dep_frame-000000_0.png", cv2.IMREAD_ANYDEPTH)    

pred_dep = pred_dep / 1000.0
raw_dep = raw_dep / 1000.0
valid_pos = np.where(raw_dep>0)

pred_dep_valid = pred_dep[valid_pos].flatten()
raw_dep_valid = raw_dep[valid_pos].flatten()

mse = np.mean(np.square(pred_dep_valid - raw_dep_valid))
print(mse)