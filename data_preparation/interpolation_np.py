import os
import cv2
import numpy as np
import time


def interpolation_kernel(img, pos):
    posx, posy = pos[1], pos[0]
    for difx in range(-2, 3):
        for dify in range(-2, 3):
            new_posx, new_posy = posx+difx, posy+dify
            if (new_posx >= 0 and new_posx < img.shape[1] and new_posy > 0 and new_posy < img.shape[0]):
                if(img[new_posy, new_posx]>0):
                    interpolation_posx, interpolation_posy = (new_posx + posx) // 2, (new_posy + posy) // 2
                    if (img[interpolation_posy, interpolation_posx] == 0):
                        # print("interpo",interpolation_posy, interpolation_posx)
                        # input()
                        img[interpolation_posy, interpolation_posx] = (img[posy, posx] + img[new_posy, new_posx]) / 2
    
        
        
PATH = "data0929/aligned_depth"
out_PATH = "data0929/interpo_depth"

depth_names = os.listdir(PATH)
depth_paths = [os.path.join(PATH, depth_name) for depth_name in depth_names]

def main():

    for i in range(len(depth_names)):
        start_t = time.time()
        name = depth_names[i]
        path = depth_paths[i]
        
        print(name)
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        img = np.asarray(img).astype(np.float32)
        
        mask = np.where(img>0)
        valid_pos = np.array(list(zip(mask[0],mask[1])))

        
        width, height = img.shape[1], img.shape[0]
        center_mask = (np.array([2]), np.array([2]))
        for pos in valid_pos:
            row, col = pos[0], pos[1]
            if row-2 >= 0 and col-2 >= 0 and row+2 < height and col+2 < width:
                kernel = img[row-2:row+3, col-2:col+3]
                kernel_cp = img[row-2:row+3, col-2:col+3].copy()
                valid_mask = np.where(kernel>0)
                interpo_mask = ((valid_mask[0] + center_mask[0]) // 2, (valid_mask[1] + center_mask[1]) // 2)

                inter_val = (kernel[valid_mask] + kernel[2,2]) / 2
                
                kernel[interpo_mask] = inter_val[:]
                kernel[valid_mask] = kernel_cp[valid_mask]
                
                
        img = img.astype(np.uint16)
        outpath = os.path.join(out_PATH, name)
        cv2.imwrite(outpath, img)
        end_t = time.time()
        print(end_t - start_t)
if __name__ == '__main__':
    main()