import os
import cv2
import numpy as np
import os

def interpolation_kernel(img, pos, size):
    posx, posy = pos[1], pos[0]

    for difx in range(-size//2, size//2):
        for dify in range(-size//2, size//2):
            new_posx, new_posy = posx+difx, posy+dify
            if (new_posx >= 0 and new_posx < img.shape[1] and new_posy > 0 and new_posy < img.shape[0]):
                if(img[new_posy, new_posx]>0):
                    interpolation_posx, interpolation_posy = (new_posx + posx) // 2, (new_posy + posy) // 2
                    if (img[interpolation_posy, interpolation_posx] == 0):
                        # print("interpo",interpolation_posy, interpolation_posx)
                        # input()
                        img[interpolation_posy, interpolation_posx] = (img[posy, posx] + img[new_posy, new_posx]) / 2
    # return img
PATH = "aligned_depth"
out_PATH = "interpo_depth"

depth_names = os.listdir(PATH)
depth_paths = [os.path.join(PATH, depth_name) for depth_name in depth_names]

for i in range(len(depth_paths)):
    name = depth_names[i]
    path = depth_paths[i]
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    mask = np.where(img>0)
    valid_pos = np.array(list(zip(mask[0],mask[1])))
    print(valid_pos)
    input()
    for pos in valid_pos:
        # print(pos)
        # input()
        interpolation_kernel(img, pos, 5)

    img = img.astype(np.uint16)
    outpath = os.path.join(out_PATH, name)
    cv2.imwrite(outpath, img)