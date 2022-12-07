import os
import cv2
import numpy as np
import time
import multiprocessing as mp

    
def inter_kernel(img):
    
    
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
    return img


PATH = "data0929/aligned_depth"
out_PATH = "data0929/interpo_depth"

depth_names = os.listdir(PATH)
depth_paths = [os.path.join(PATH, depth_name) for depth_name in depth_names]
num_cores = int(mp.cpu_count())


def main():
    start_t = time.time()
    imgs = np.zeros((16, 1080, 1920)) # (img_idx, height, width)
    names=[]
    for i in range(min(len(depth_names),16)):
        start_t = time.time()
        name = depth_names[i]
        path = depth_paths[i]
        
        print(name)
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        img = np.asarray(img).astype(np.float32)
        imgs[i] = img
        names.append(name)
    
    pool = mp.Pool(num_cores)
    results = [pool.apply_async(inter_kernel, args=[img]) for img in imgs]
    results = [p.get() for p in results]
    for i in range(len(imgs)):
        img = results[i]
        name = names[i]
        img = img.astype(np.uint16)
        outpath = os.path.join(out_PATH, name)
        cv2.imwrite(outpath, img)
    end_t = time.time()
    print(end_t - start_t)
    

if __name__ == '__main__':
    main()