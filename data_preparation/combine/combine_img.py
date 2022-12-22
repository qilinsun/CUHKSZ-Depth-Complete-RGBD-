import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import os

DATASET = 'data0929'

PATH = "gallary/eval_gallary/data0929/cost_spn_epoch2"
out_PATH = "gallary/eval_gallary/data0929/cost_spn_epoch2/combine"

if not os.path.exists(PATH):
    os.makedirs(PATH)   

if not os.path.exists(out_PATH):
    os.makedirs(out_PATH)

if DATASET == 'data0929':
    raw_depth_names = os.listdir(os.path.join(PATH, 'scale_raw'))
    raw_depth_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
    raw_depth_paths = [os.path.join(PATH, 'scale_raw', depth_name) for depth_name in raw_depth_names]
    
else:
    raw_depth_names = os.listdir(os.path.join(PATH, 'unscale_raw'))
    raw_depth_names.sort(key=lambda x: int(x.split("_")[5][1:x.split("_")[5].find('.')]))
    raw_depth_paths = [os.path.join(PATH, 'unscale_raw', depth_name) for depth_name in raw_depth_names]


if DATASET == 'data0929':
    pred_depth_names = os.listdir(os.path.join(PATH, 'scale_pred'))
    pred_depth_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
    pred_depth_paths = [os.path.join(PATH, 'scale_pred', depth_name) for depth_name in pred_depth_names]
    
else:
    pred_depth_names = os.listdir(os.path.join(PATH, 'unscale_pred'))
    pred_depth_names.sort(key=lambda x: int(x.split("_")[5][1:x.split("_")[5].find('.')]))
    pred_depth_paths = [os.path.join(PATH, 'unscale_pred', depth_name) for depth_name in pred_depth_names]

color_names = os.listdir(os.path.join(PATH, 'color'))
if DATASET == 'data0929':
    color_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
else:
    color_names.sort(key=lambda x: int(x.split("_")[4][1:x.split("_")[4].find('.')]))

color_paths = [os.path.join(PATH, 'color', color_name) for color_name in color_names]


for i in range(len(color_names)):
    _, figs = plt.subplots(1, 3, figsize=(200, 40))

    raw_name = raw_depth_names[i]
    pred_name = pred_depth_names[i]
    color_name = color_names[i]
    
    raw_path = raw_depth_paths[i]
    pred_path = pred_depth_paths[i]
    color_path = color_paths[i]

    bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    rgb = bgr[...,::-1]
    
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    
    figs[0].imshow(rgb)
    figs[0].set_title('rgb_image')
    figs[1].imshow(raw)
    figs[1].set_title('raw_depth')
    figs[2].imshow(pred)
    figs[2].set_title('pred_depth')
    
    
    result_name = color_name[color_name.find("_")+1:]
    plt.savefig(os.path.join(out_PATH, result_name))
    
    
    