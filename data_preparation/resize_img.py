import os
import cv2
import numpy as np
from Camera import Camera
import time
import operator
import argparse
from PIL import Image




if __name__ == '__main__':
    
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Format Converter - PGM')

    ### Positional arguments

    ### Optional arguments

    parser.add_argument('--rgb_input_dir', default='data_0929/cropped_rgb', type=str, help='Path to the pgm file')
    parser.add_argument('--rgb_output_dir', default='data_0929/resize_rgb', type=str, help='Path to the result file')
    parser.add_argument('--dep_input_dir', default='data_0929/cropped_interpo_dep', type=str, help='Dir to the pgm files')
    parser.add_argument('--dep_output_dir', default='data_0929/resize_interpo_dep', type=str, help='Dir to the result files')
    parser.add_argument('-b', '--batch', action="store_true", default=True, help='Batch processing') 

    args = vars(parser.parse_args())
    # print(args)
    
    dep_PATH = args['dep_input_dir']
    out_dep_PATH =  args['dep_output_dir']
    rgb_PATH =  args['rgb_input_dir']
    out_rgb_PATH = args['rgb_output_dir']

    if not os.path.exists(out_rgb_PATH):
        os.makedirs(out_rgb_PATH)
        
    if not os.path.exists(out_dep_PATH):
        os.makedirs(out_dep_PATH)
        
    depth_names = os.listdir(dep_PATH)
    rgb_names = os.listdir(rgb_PATH)

    depth_paths = [os.path.join(dep_PATH, depth_name) for depth_name in depth_names]
    rgb_paths = [os.path.join(rgb_PATH, rgb_name) for rgb_name in rgb_names]
    
    for i in range(len(depth_paths)):
        dep_name = depth_names[i]
        rgb_name = rgb_names[i]
        dep_path = depth_paths[i]
        rgb_path = rgb_paths[i]

        rgb = cv2.imread(rgb_path)
        dep = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)
        
        resize_rgb = cv2.resize(rgb, (320,256), cv2.INTER_NEAREST)
        resize_dep = cv2.resize(dep, (320,256), cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(out_rgb_PATH, rgb_name), resize_rgb)
        cv2.imwrite(os.path.join(out_dep_PATH, dep_name), resize_dep)