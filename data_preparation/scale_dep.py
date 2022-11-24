import os
import cv2
import numpy as np
import os

PATH = "depth"
out_PATH = "origin_scale_dep"

depth_names = os.listdir(PATH)
depth_paths = [os.path.join(PATH, depth_name) for depth_name in depth_names]

for i in range(len(depth_paths)):
    name = depth_names[i]
    path = depth_paths[i]
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    
    img = img * 2**16 / np.max(img)
    
    img = img.astype(np.uint16)
    outpath = os.path.join(out_PATH, name)
    cv2.imwrite(outpath, img)