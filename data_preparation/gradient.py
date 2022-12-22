import cv2
import os

PATH = './report/batchsize_16_gradientloss/matterport/color'

OUT_PATH = './data_preparation/test'

img = cv2.imread(os.path.join(PATH, 'color_fzynW3qQPVF_23f90479f2cf4c60bc78cb3252fe64e8_d1_1.png'), cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)



cv2.imwrite(os.path.join(OUT_PATH, 'color_sobelx.png'), sobelx)
cv2.imwrite(os.path.join(OUT_PATH, 'color_sobely.png'), sobely)