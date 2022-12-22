import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import os

PATH = "gallary/eval_gallary/l1gradient_loss"
out_PATH = "gallary/eval_gallary/l1gradient_loss/combine"
# PATH = "./report/sigloss/matterport"
# out_PATH = "./report/silog_loss/matterport/combine"

color_names = os.listdir(os.path.join(PATH, 'color'))
color_names.sort(key=lambda x: x.split("_")[2]+x.split("_")[4])
color_paths = [os.path.join(PATH, 'color', color_name) for color_name in color_names]

gt_names = os.listdir(os.path.join(PATH, 'unscale_gt'))
gt_names.sort(key=lambda x: x.split("_")[3]+x.split("_")[5])
gt_paths = [os.path.join(PATH, 'unscale_gt', depth_name) for depth_name in gt_names]
    
pred_depth_names = os.listdir(os.path.join(PATH, 'unscale_pred'))
pred_depth_names.sort(key=lambda x: x.split("_")[3]+x.split("_")[5])
pred_depth_paths = [os.path.join(PATH, 'unscale_pred', depth_name) for depth_name in pred_depth_names]

raw_depth_names = os.listdir(os.path.join(PATH, 'unscale_raw'))
raw_depth_names.sort(key=lambda x: x.split("_")[3]+x.split("_")[5])
raw_depth_paths = [os.path.join(PATH, 'unscale_raw', depth_name) for depth_name in raw_depth_names]

for i in range(0, len(color_names), 10):
    _, figs = plt.subplots(1, 4, figsize=(150,30))

    raw_name = raw_depth_names[i]
    pred_name = pred_depth_names[i]
    color_name = color_names[i]
    gt_name = gt_names[i]
    
    raw_path = raw_depth_paths[i]
    pred_path = pred_depth_paths[i]
    color_path = color_paths[i]
    gt_path = gt_paths[i]

    bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    rgb = bgr[...,::-1]
    
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    figs[0].imshow(rgb)
    figs[0].set_title('rgb image', loc='center')
    figs[1].imshow(raw)
    figs[1].set_title('raw depth image')
    
    figs[2].imshow(pred)
    figs[2].set_title('pred depth image')
    
    figs[3].imshow(gt)
    figs[3].set_title('gt depth image')
    
    result_name = color_name[color_name.find("_")+1:]
    plt.savefig(os.path.join(out_PATH, result_name))
    
    
    