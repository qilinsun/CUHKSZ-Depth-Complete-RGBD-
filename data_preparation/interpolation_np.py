import os
import cv2
import numpy as np
import time

def interpolation_kernel(img, pos, size):
    # posx, posy = pos[1], pos[0]
    # for difx in range(-size//2, size//2):
    #     for dify in range(-size//2, size//2):
    #         new_posx, new_posy = posx+difx, posy+dify
    #         if (new_posx >= 0 and new_posx < img.shape[1] and new_posy > 0 and new_posy < img.shape[0]):
    #             if(img[new_posy, new_posx]>0):
    #                 interpolation_posx, interpolation_posy = (new_posx + posx) // 2, (new_posy + posy) // 2
    #                 if (img[interpolation_posy, interpolation_posx] == 0):
    #                     # print("interpo",interpolation_posy, interpolation_posx)
    #                     # input()
    #                     img[interpolation_posy, interpolation_posx] = (img[posy, posx] + img[new_posy, new_posx]) / 2
    
    row, col = pos[0], pos[1]
    width, height = img.shape[1], img.shape[0]
    center = np.array([2,2])
    if row-2 >= 0 and col-2 >= 0 and row+2 < height and col+2 < width:
        kernel = img[row-2:row+3, col-2:col+3]
        mask = np.where(kernel>0)
        valid_pos = np.array(list(zip(mask[0],mask[1])))
        inter_pos = np.array([(v + center) // 2 for v in valid_pos])
        inter_val = (kernel[mask] + kernel[2,2]) / 2
        for i in range(len(inter_pos)):
            r, c = inter_pos[i][0], inter_pos[i][1]
            img[row+r-2, col+c-2] = inter_val[i]
            
        
        
PATH = "data0929/aligned_depth"
out_PATH = "data0929/interpo_depth"

depth_names = os.listdir(PATH)
depth_paths = [os.path.join(PATH, depth_name) for depth_name in depth_names]

for i in range(len(depth_paths)):
    start_t = time.time()
    name = depth_names[i]
    path = depth_paths[i]
    
    print(name)
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    img = np.asarray(img).astype(np.float32)
    # print(img.dtype)
    mask = np.where(img>0)
    valid_pos = np.array(list(zip(mask[0],mask[1])))
    # print(len(valid_pos))
    # input()
    
    width, height = img.shape[1], img.shape[0]
    center = np.array([2,2])
    
    for pos in valid_pos:
        # interpolation_kernel(img, pos, 5)
        row, col = pos[0], pos[1]
        if row-2 >= 0 and col-2 >= 0 and row+2 < height and col+2 < width:
            start_t = time.time()
            kernel = img[row-2:row+3, col-2:col+3]
            mask = np.where(kernel>0)
            valid_pos = np.array(list(zip(mask[0],mask[1])))
            inter_pos = (valid_pos + center) // 2
            inter_val = (kernel[mask] + kernel[2,2]) / 2
            for i in range(len(inter_pos)):
                r, c = inter_pos[i][0], inter_pos[i][1]
                img[row+r-2, col+c-2] = inter_val[i]
            end_t = time.time()
            print(end_t - start_t)
            
    img = img.astype(np.uint16)
    outpath = os.path.join(out_PATH, name)
    cv2.imwrite(outpath, img)
    end_t = time.time()
    print(end_t - start_t)