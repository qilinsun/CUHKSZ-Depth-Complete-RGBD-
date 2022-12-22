import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import os

DATASET = 'data0929'

PATH1 = "gallary/eval_gallary/data_0929"
PATH2 = "report/batchsize_16_gradientloss/ourdata"

out_PATH = "report/data0929_1219_compare"

if not os.path.exists(out_PATH):
    os.makedirs(out_PATH)
    
if DATASET == 'data0929':
    raw_depth_names = os.listdir(os.path.join(PATH1, 'scale_raw'))
    raw_depth_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
    raw_depth_paths = [os.path.join(PATH1, 'scale_raw', depth_name) for depth_name in raw_depth_names]
    
else:
    raw_depth_names = os.listdir(os.path.join(PATH1, 'unscale_raw'))
    raw_depth_names.sort(key=lambda x: int(x.split("_")[5][1:x.split("_")[5].find('.')]))
    raw_depth_paths = [os.path.join(PATH1, 'unscale_raw', depth_name) for depth_name in raw_depth_names]

if DATASET == 'data0929':
    pred_depth_names1 = os.listdir(os.path.join(PATH1, 'scale_pred'))
    pred_depth_names1.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
    pred_depth_paths1 = [os.path.join(PATH1, 'scale_pred', depth_name) for depth_name in pred_depth_names1]
    
else:
    pred_depth_names1 = os.listdir(os.path.join(PATH1, 'unscale_raw'))
    pred_depth_names1.sort(key=lambda x: int(x.split("_")[5][1:x.split("_")[5].find('.')]))
    pred_depth_paths1 = [os.path.join(PATH1, 'unscale_raw', depth_name) for depth_name in pred_depth_names1]


if DATASET == 'data0929':
    pred_depth_names2 = os.listdir(os.path.join(PATH2, 'scale_pred'))
    pred_depth_names2.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
    pred_depth_paths2 = [os.path.join(PATH2, 'scale_pred', depth_name) for depth_name in pred_depth_names2]
    
else:
    pred_depth_names2 = os.listdir(os.path.join(PATH2, 'unscale_pred'))
    pred_depth_names2.sort(key=lambda x: int(x.split("_")[5][1:x.split("_")[5].find('.')]))
    pred_depth_paths2 = [os.path.join(PATH2, 'unscale_pred', depth_name) for depth_name in pred_depth_names2]

color_names = os.listdir(os.path.join(PATH2, 'color'))
if DATASET == 'data0929':
    color_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
else:
    color_names.sort(key=lambda x: int(x.split("_")[4][1:x.split("_")[4].find('.')]))

color_paths = [os.path.join(PATH1, 'color', color_name) for color_name in color_names]


for i in range(len(color_names)):
    _, figs = plt.subplots(1, 4, figsize=(200, 40))

    raw_name = raw_depth_names[i]
    pred_name1 = pred_depth_names1[i]
    pred_name2 = pred_depth_names2[i]
    color_name = color_names[i]
    
    raw_path = raw_depth_paths[i]
    pred_path1 = pred_depth_paths1[i]
    pred_path2 = pred_depth_paths2[i]
    color_path = color_paths[i]

    bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    rgb = bgr[...,::-1]
    
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    pred1 = cv2.imread(pred_path1, cv2.IMREAD_GRAYSCALE)
    pred2 = cv2.imread(pred_path2, cv2.IMREAD_GRAYSCALE)
    
    figs[0].imshow(rgb)
    figs[0].set_title('rgb_image')
    figs[1].imshow(raw)
    figs[1].set_title('raw_depth')
    figs[2].imshow(pred1)
    figs[2].set_title('pred_depth')
    figs[3].imshow(pred2)
    figs[3].set_title('pred_depth')
    
    result_name = color_name[color_name.find("_")+1:]
    plt.savefig(os.path.join(out_PATH, result_name))
    
    
    