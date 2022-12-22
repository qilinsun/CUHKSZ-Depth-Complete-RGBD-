import os
import cv2
import numpy as np
import os

PATH = "gallary/eval_gallary/data0929/cost_spn_epoch2"
out_PATH = "gallary/eval_gallary/data0929/cost_spn_epoch2"

raw_depth_names = os.listdir(os.path.join(PATH, 'unscale_raw'))
raw_depth_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
raw_depth_paths = [os.path.join(PATH, 'unscale_raw', depth_name) for depth_name in raw_depth_names]

pred_depth_names = os.listdir(os.path.join(PATH, 'unscale_pred'))
pred_depth_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
pred_depth_paths = [os.path.join(PATH, 'unscale_pred', depth_name) for depth_name in pred_depth_names]

pred_out_path = os.path.join(out_PATH, 'scale_pred')
raw_out_path = os.path.join(out_PATH, 'scale_raw')

if not os.path.exists(pred_out_path):
    os.makedirs(pred_out_path)
    
if not os.path.exists(raw_out_path):
    os.makedirs(raw_out_path)
    
for i in range(len(raw_depth_names)):
    
    raw_name = raw_depth_names[i]
    pred_name = pred_depth_names[i]
    
    raw_path = raw_depth_paths[i]
    pred_path = pred_depth_paths[i]
    
    pred_dep = cv2.imread(pred_path, cv2.IMREAD_ANYDEPTH)    
    raw_dep = cv2.imread(raw_path, cv2.IMREAD_ANYDEPTH)    
    
    maximum = max(np.max(raw_dep), np.max(pred_dep))
    
    raw = raw_dep * 2**16 / np.max(pred_dep)
    pred_dep = pred_dep * 2**16 / np.max(pred_dep)
    
    raw = raw.astype(np.uint16)
    pred_dep = pred_dep.astype(np.uint16)
    
    raw_outpath = os.path.join(os.path.join(out_PATH, 'scale_raw'), raw_name)
    pred_outpath = os.path.join(os.path.join(out_PATH, 'scale_pred'), pred_name)
    
    
    cv2.imwrite(raw_outpath, raw)
    cv2.imwrite(pred_outpath, pred_dep)