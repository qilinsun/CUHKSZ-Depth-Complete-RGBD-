import cv2
import numpy as np
import Camera
import time
import os
import operator
import argparse
from PIL import Image

camera = Camera.Camera()

def undistort(in_dir, out_dir):
    file_list = os.listdir(in_dir)
    for file_name in file_list:
        file_path = os.path.join(in_dir, file_name)
        out_path = os.path.join(out_dir, file_name)
        img = Image.open(file_path)
        img = np.array(img, dtype=np.uint16)
        if file_name.split('_')[2] == "Depth":
            K = camera.depth_intrinsic
            dist_coeffs = camera.depth_distortion
        elif file_name.split('_')[2] == "RGB":
            K = camera.rgb_intrinsic
            dist_coeffs = camera.rgb_distortion
            
        img_undistored = cv2.undistort(img, K, dist_coeffs)
        
        cv2.imwrite(out_path, img_undistored)


if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Format Converter - PGM')

    ### Positional arguments

    ### Optional arguments

    parser.add_argument('-i', '--input', type=str, help='Path to the pgm file')
    parser.add_argument('-o', '--output', type=str, help='Path to the result file')
    parser.add_argument('--input_dir', default='data_1219/lib/origin_raw_dep', type=str, help='Dir to the pgm files')
    parser.add_argument('--output_dir', default='data_1219/lib/undistorted_raw_dep', type=str, help='Dir to the result files')
    parser.add_argument('-b', '--batch', action="store_true", default=True, help='Batch processing') 

    args = vars(parser.parse_args())
    # print(args)
    in_path = args['input']
    out_path = args['output']

    isbatch = args['batch']
    in_dir = args['input_dir']
    out_dir = args['output_dir']

    if in_path is not None and out_path is not None:
        undistort(in_path, out_path)
    elif isbatch:
        undistort(in_dir, out_dir)
    else:
        print('请输入相应参数')

    print('Script took %s seconds.' % (time.time() - script_start_time,))