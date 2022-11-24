import os
import cv2
import numpy as np
from Camera import Camera

def depth_rgb_align(depth_path, align_depth_path, rgb_path, aligned_depth_rgb_path):

    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) # depth in mm
    aligned_depth = np.zeros((1080, 1920))
    aligned_depth_rgb = cv2.imread(rgb_path)
    # depth = depth/1000.0
    # print(np.max(depth))

    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            dep = depth[i, j] 
            if (dep != 0):
                p_dep = np.array([j, i, 1])
                p_dep = p_dep * dep # projected pixel homegeneous coord of TOF camera plane.
                P_dep_3d = np.dot(np.linalg.inv(camera_info.depth_intrinsic), p_dep) # 3d point in TOF camera space
                # R*P_rgb + T = P_dep_3d. -> P_rgb = (R-1)*P_dep_3d - R-1*T
                P_rgb_3d = np.dot(camera_info.rotation_matrix, P_dep_3d) + camera_info.translation_matrix# Transform d3 point from depth camera space to rgb cam space.
                p_rgb_2d = np.dot(camera_info.rgb_intrinsic, P_rgb_3d) #   the 3d point into rgb cameara plane
                dep_rgb = P_rgb_3d[2]

                p_rgb_2d = p_rgb_2d / dep_rgb # Note that p_rgb_2d is homegeneous, it's necessary to divede it by the depth.
                p_rgb_x, p_rgb_y = int(p_rgb_2d[0]), int(p_rgb_2d[1])
                if (p_rgb_x < 1920 and p_rgb_x >= 0 and p_rgb_y < 1080 and p_rgb_y >= 0):
                    aligned_depth[p_rgb_y, p_rgb_x] = dep_rgb

    # aligned_depth = aligned_depth*1000.0
    aligned_depth_rgb[np.where(aligned_depth == 0)] = 0
    aligned_depth = aligned_depth.astype(np.uint16)
    # aligned_depth_rgb = aligned_depth_rgb.astype(np.uint16)
    cv2.imwrite(align_depth_path, aligned_depth)
    cv2.imwrite(aligned_depth_rgb_path, aligned_depth_rgb)

def undistort(img, intrinsic, distortion):
    h, w = img.shape[0], img.shape[1]
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsic)


if __name__ == '__main__':
    
    depth_folder = "depth"
    aligned_depth_folder =  "aligned_depth"
    rgb_folder =  "origin_rgb"
    aligned_depth_rgb_folder = "aligned_depth_rgb"

    depth_names = os.listdir(depth_folder)
    rgb_names = os.listdir(rgb_folder)

    depth_names.sort()
    rgb_names.sort()
    # print(depth_names)
    
    depth_paths = [os.path.join(depth_folder, depth_name) for depth_name in depth_names]
    rgb_paths = [os.path.join(rgb_folder, rgb_name) for rgb_name in rgb_names]

    aligned_depth_paths = [os.path.join(aligned_depth_folder, depth_name) for depth_name in depth_names]
    aligned_depth_rgb_paths = [os.path.join(aligned_depth_rgb_folder, rgb_name) for rgb_name in rgb_names]

    # print(aligned_depth_paths)
    camera_info = Camera()

    for i in range(len(depth_paths)):
        depth_rgb_align(depth_paths[i], aligned_depth_paths[i], rgb_paths[i], aligned_depth_rgb_paths[i])

    for i in range(len(depth_paths)):
        aligned_dep = cv2.imread(aligned_depth_paths[i], cv2.IMREAD_ANYDEPTH)
        dep = cv2.imread(depth_paths[i], cv2.IMREAD_ANYDEPTH)
        print("align:" ,np.max(aligned_dep))
        print("dep: :", np.max(dep))
