import os
import cv2
import numpy as np
import time
import multiprocessing as mp

    
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


PATH = "data0929/aligned_depth"
out_PATH = "data0929/interpo_depth"

depth_names = os.listdir(PATH)
num_img = len(depth_names)
depth_paths = [os.path.join(PATH, depth_name) for depth_name in depth_names]
num_cores = 32


def main():
    start_t = time.time()
    imgs = np.zeros((num_cores, 1080, 1920)) # (img_idx, height, width)
    names=[None] * num_cores
    for j in range(num_img // num_cores + (num_img % num_cores != 0)):
        num_img_to_process = min(num_cores*(j+1), num_img) - num_cores*j
        for i in range(num_cores*j, min(num_cores*(j+1), num_img)):
            name = depth_names[i]
            path = depth_paths[i]
            
            print(name)
            img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            img = np.asarray(img).astype(np.float32)
            imgs[i%num_cores] = img
            names[i%num_cores] = name
        
        pool = mp.Pool(num_cores)
        results = [pool.apply_async(inter_kernel, args=[imgs[i], 7]) for i in range(num_img_to_process)]
        results = [p.get() for p in results]
    
        for i in range(min(num_img, num_cores)):
            img = results[i]
            name = names[i]
            img = img.astype(np.uint16)
            outpath = os.path.join(out_PATH, name)
            cv2.imwrite(outpath, img)
    end_t = time.time()
    print(end_t - start_t)
    

if __name__ == '__main__':
    main()