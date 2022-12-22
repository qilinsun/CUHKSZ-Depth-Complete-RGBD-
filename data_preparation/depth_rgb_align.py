import os
import cv2
import numpy as np
from Camera import Camera
import time
import operator
import argparse
from PIL import Image

def depth_rgb_align(depth_path, align_depth_path):

    depth = Image.open(depth_path) # depth in mm
    depth = np.array(depth, dtype=np.uint16)
    aligned_depth = np.zeros((1080, 1920))
    
    flatten_dep = np.expand_dims(depth.flatten(), axis=1) # N * 1 depth array
    xy_pos = np.array([[i % depth.shape[1], i // depth.shape[1]] for i in range(len(flatten_dep))])
    xyz_pos = np.concatenate((xy_pos, np.ones((len(flatten_dep), 1))), axis=1) # N * 3 array
    valid_pos = (flatten_dep > 0).squeeze()
    xyz_pos = xyz_pos[valid_pos]
    flatten_dep = flatten_dep[flatten_dep>0]
    p_dep = xyz_pos * np.expand_dims(flatten_dep, axis=1)
    
    P_dep_3d = np.dot(np.linalg.inv(camera_info.depth_intrinsic), p_dep.T).T 
    h_P_dep_3d = np.concatenate((P_dep_3d, np.ones((len(flatten_dep), 1))), axis=1) # N*4 homegeneous coordinates

    h_P_rgb_3d = np.dot(camera_info.extrinsic, h_P_dep_3d.T).T
    P_rgb_3d = h_P_rgb_3d[:, :3]
    p_rgb_2d = np.dot(camera_info.rgb_intrinsic, P_rgb_3d.T).T
    
    dep_rgb = p_rgb_2d[:, 2]
    dep_rgb = dep_rgb[(dep_rgb > 0)]
    p_rgb_2d = p_rgb_2d[(dep_rgb > 0)]
    
    p_rgb_2d = p_rgb_2d / np.expand_dims(dep_rgb, axis=1)

    pos = np.array(p_rgb_2d[:, :2], dtype=np.int32)
    
    dep_rgb = dep_rgb[((pos[:, 0] >= 0) & (pos[:, 0] < 1920) & (pos[:, 1] >= 0) & (pos[:, 1] < 1080))]
    pos = pos[((pos[:, 0] >= 0) & (pos[:, 0] < 1920) & (pos[:, 1] >= 0) & (pos[:, 1] < 1080))]
    
    aligned_depth[pos[:, 1], pos[:, 0]] = dep_rgb[:]
    
    aligned_depth = aligned_depth.astype(np.uint16)

    cv2.imwrite(align_depth_path, aligned_depth)
    

if __name__ == '__main__':
    
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Format Converter - PGM')

    ### Positional arguments


    parser.add_argument('--input_dir', default='data_1219/dormitory/undistorted_raw_dep', type=str, help='Dir to the pgm files')
    parser.add_argument('--output_dir', default='data_1219/dormitory/test_aligned_raw_dep', type=str, help='Dir to the result files')
    parser.add_argument('-b', '--batch', action="store_true", default=True, help='Batch processing') 

    args = vars(parser.parse_args())
    # print(args)
    
    depth_folder = args['input_dir']
    aligned_depth_folder =  args['output_dir']

    depth_names = os.listdir(depth_folder)

    depth_names.sort()
    # print(depth_names)
    
    depth_paths = [os.path.join(depth_folder, depth_name) for depth_name in depth_names]

    aligned_depth_paths = [os.path.join(aligned_depth_folder, depth_name) for depth_name in depth_names]

    # print(aligned_depth_paths)
    camera_info = Camera()

    for i in range(len(depth_paths)):
        depth_rgb_align(depth_paths[i], aligned_depth_paths[i])

    for i in range(len(depth_paths)):
        aligned_dep = cv2.imread(aligned_depth_paths[i], cv2.IMREAD_ANYDEPTH)
        dep = cv2.imread(depth_paths[i], cv2.IMREAD_ANYDEPTH)
        
    print('Script took %s seconds.' % (time.time() - script_start_time,))