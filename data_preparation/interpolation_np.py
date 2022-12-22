import os
import cv2
import numpy as np
from Camera import Camera
import time
import operator
import argparse
from PIL import Image
import multiprocessing as mp
import statistics

def inter_kernel(img, kernel_size):
    
    r = kernel_size // 2
    mask = np.where(img>0)
    
    valid_pos = np.array(list(zip(mask[0],mask[1])))


    width, height = img.shape[1], img.shape[0]
    center_mask = (np.array([r]), np.array([r]))


    for pos in valid_pos:
        row, col = pos[0], pos[1]
        if row-r >= 0 and col-r >= 0 and row+r < height and col+r < width:
            kernel = img[row-r:row+r+1, col-r:col+r+1]
            kernel_cp = img[row-r:row+r+1, col-r:col+r+1].copy()
            valid_mask = np.where(kernel>0)
            interpo_mask = ((valid_mask[0] + center_mask[0]) // 2, (valid_mask[1] + center_mask[1]) // 2)

            inter_val = (kernel[valid_mask] + kernel[r,r]) / 2
            
            kernel[interpo_mask] = inter_val[:]
            kernel[valid_mask] = kernel_cp[valid_mask]
    return img

def nn_inter_kernel(img, kernel_size):
    
    r = kernel_size // 2
    mask = np.where(img>0)
    
    valid_pos = np.array(list(zip(mask[0],mask[1])))
    width, height = img.shape[1], img.shape[0]
    center_mask = (np.array([r]), np.array([r]))

    for pos in valid_pos:
        row, col = pos[0], pos[1]
        if row-r >= 0 and col-r >= 0 and row+r < height and col+r < width:
            kernel = img[row-r:row+r+1, col-r:col+r+1]
            kernel_cp = img[row-r:row+r+1, col-r:col+r+1].copy()
            valid_mask = np.where(kernel>0)
            interpo_mask = ((valid_mask[0] + center_mask[0]) // 2, (valid_mask[1] + center_mask[1]) // 2)

            inter_val = kernel[r,r]
            
            kernel[interpo_mask] = inter_val
            kernel[valid_mask] = kernel_cp[valid_mask]

    return img

if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Format Converter - PGM')

    ### Positional arguments

    ### Optional arguments

    parser.add_argument('-i', '--input', type=str, help='Path to the pgm file')
    parser.add_argument('-o', '--output', type=str, help='Path to the result file')
    parser.add_argument('--input_dir', default='data_1219/dormitory/aligned_raw_dep', type=str, help='Dir to the pgm files')
    parser.add_argument('--output_dir', default='data_1219/dormitory/interpo_depth', type=str, help='Dir to the result files')
    parser.add_argument('-b', '--batch', action="store_true", default=True, help='Batch processing') 

    args = vars(parser.parse_args())
    # print(args)
    in_path = args['input']
    out_path = args['output']

    isbatch = args['batch']
    in_dir = args['input_dir']
    out_dir = args['output_dir']
    

    depth_names = os.listdir(in_dir)
    num_img = len(depth_names)
    depth_paths = [os.path.join(in_dir, depth_name) for depth_name in depth_names]
    num_cores = 32

    imgs = np.zeros((num_cores, 1080, 1920)) # (img_idx, height, width)
    names=[None] * num_cores
    for j in range(num_img // num_cores + (num_img % num_cores != 0)):
        num_img_to_process = min(num_cores*(j+1), num_img) - num_cores*j
        for i in range(num_cores*j, min(num_cores*(j+1), num_img)):
            name = depth_names[i]
            path = depth_paths[i]
            
            img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            img = np.asarray(img).astype(np.float32)
            imgs[i%num_cores] = img
            names[i%num_cores] = name
        
        pool = mp.Pool(num_cores)
        results = [pool.apply_async(nn_inter_kernel, args=[imgs[i], 7]) for i in range(num_img_to_process)]
        results = [p.get() for p in results]
    
        for i in range(min(num_img, num_cores)):
            img = results[i]
            name = names[i]
            img = img.astype(np.uint16)
            outpath = os.path.join(out_dir, name)
            cv2.imwrite(outpath, img)

    print('Script took %s seconds.' % (time.time() - script_start_time,))