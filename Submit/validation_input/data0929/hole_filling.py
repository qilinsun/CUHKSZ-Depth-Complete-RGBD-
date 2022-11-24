
# Import modules
from importlib.abc import FileLoader
from symbol import star_expr
import cv2
import numpy as np
import math
import sys, time
import os


# path = "EMDC-PyTorch/Submit/validation_input/data0929/aligned_depth"
path = "bicubic_aligned_depth"
img_names = os.listdir(path)



# Interpolation kernel
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Paddnig
def padding(img,H,W,C):
    zimg = np.zeros((H+4,W+4,C))
    zimg[2:H+2,2:W+2,:C] = img
    #Pad the first/last two col and row
    zimg[2:H+2,0:2,:C]=img[:,0:1,:C]
    zimg[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]
    zimg[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]
    zimg[0:2,2:W+2,:C]=img[0:1,:,:C]
    #Pad the missing eight points
    zimg[0:2,0:2,:C]=img[0,0,:C]
    zimg[H+2:H+4,0:2,:C]=img[H-1,0,:C]
    zimg[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]
    zimg[0:2,W+2:W+4,:C]=img[0,W-1,:C]
    return zimg

# Bicubic operation
def bicubic(img, ratio, a):
    #Get image size
    if (len(img.shape) == 2):
        img = np.expand_dims(img, 2)
    H,W,C = img.shape

    img = padding(img,H,W,C)
    #Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    dst = np.zeros((dH, dW, C))

    h = 1/ratio

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                if (img[j,i,c] == 0):
                    x, y = i * h + 2 , j * h + 2

                    x1 = 1 + x - math.floor(x)
                    x2 = x - math.floor(x)
                    x3 = math.floor(x) + 1 - x
                    x4 = math.floor(x) + 2 - x

                    y1 = 1 + y - math.floor(y)
                    y2 = y - math.floor(y)
                    y3 = math.floor(y) + 1 - y
                    y4 = math.floor(y) + 2 - y

                    mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                    mat_m = np.matrix([[img[int(y-y1),int(x-x1),c],img[int(y-y2),int(x-x1),c],img[int(y+y3),int(x-x1),c],img[int(y+y4),int(x-x1),c]],
                                    [img[int(y-y1),int(x-x2),c],img[int(y-y2),int(x-x2),c],img[int(y+y3),int(x-x2),c],img[int(y+y4),int(x-x2),c]],
                                    [img[int(y-y1),int(x+x3),c],img[int(y-y2),int(x+x3),c],img[int(y+y3),int(x+x3),c],img[int(y+y4),int(x+x3),c]],
                                    [img[int(y-y1),int(x+x4),c],img[int(y-y2),int(x+x4),c],img[int(y+y3),int(x+x4),c],img[int(y+y4),int(x+x4),c]]])
                    mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
                    dst[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)
                else:
                    dst[j,i,c] = img[j,i,c]
    return dst


for name in img_names:
    start_time = time.time()
    img_path = os.path.join(path, name)

    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)

    print("processing", name)
    filled_img = bicubic(img, 1, 0.5)
    filled_img = filled_img.astype(np.uint16)
    dep_path = os.path.join("bicubic_aligned_depth", name)
    cv2.imwrite(dep_path, filled_img)
    end_time = time.time()
    print("it takes", end_time - start_time)
