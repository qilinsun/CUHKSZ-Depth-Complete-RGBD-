import cv2
import os.path
import glob
import numpy as np
from PIL import Image

path = "new_max_aligned_depth"
img_names = os.listdir(path)


def convertPNG(im_depth,outpath):
    #读取16位深度图（像素范围0～65535），并将其转化为8位（像素范围0～255）
    # uint16_img = cv2.imread(im_depth, -1)    #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取，否则默认的读取类型为8位。
    im_depth -= im_depth.min()
    im_depth = im_depth / (im_depth.max() - im_depth.min())
    im_depth *= 255
    #使得越近的地方深度值越大，越远的地方深度值越小，以达到伪彩色图近蓝远红的目的。
    im_depth = 255 - im_depth
    #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=1),cv2.COLORMAP_JET)
    #convert to mat png
    im=Image.fromarray(im_color)
    #save image
    im.save(outpath)
 
for name in img_names:
    if name[-7:-4] != "jet":
        img_path = os.path.join(path, name)

        img = cv2.imread(img_path, -1)
        # img = img / 1000
        outpath = os.path.join(path, "color"+name)
        convertPNG(img,outpath)