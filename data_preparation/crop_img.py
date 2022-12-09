import os
import cv2
import numpy as np


rgb_PATH = "data0929/origin_rgb"
dep_PATH = "data0929/interpo_depth"

out_rgb_PATH = "data0929/cropped_rgb"
out_dep_PATH = "data0929/cropped_interpo_depth"

if not os.path.exists(out_rgb_PATH):
    os.makedirs(out_rgb_PATH)
    
if not os.path.exists(out_dep_PATH):
    os.makedirs(out_dep_PATH)
    
depth_names = os.listdir(dep_PATH)
rgb_names = os.listdir(rgb_PATH)

depth_paths = [os.path.join(dep_PATH, depth_name) for depth_name in depth_names]
rgb_paths = [os.path.join(rgb_PATH, rgb_name) for rgb_name in rgb_names]

w, h = 1080, 864
for i in range(len(depth_paths)):
    dep_name = depth_names[i]
    rgb_name = rgb_names[i]
    dep_path = depth_paths[i]
    rgb_path = rgb_paths[i]

    rgb = cv2.imread(rgb_path)
    dep = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)

    center = (rgb.shape[0] // 2, rgb.shape[1] // 2)
    x = center[1] - w//2
    y = center[0] - h//2
    
    crop_rgb = rgb[y:y+h, x:x+w]
    crop_dep = dep[y:y+h, x:x+w]
    
    cv2.imwrite(os.path.join(out_rgb_PATH, rgb_name), crop_rgb)
    cv2.imwrite(os.path.join(out_dep_PATH, dep_name), crop_dep)