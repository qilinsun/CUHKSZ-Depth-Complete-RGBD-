import os
import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import math
from PIL import Image
PATH = "gallary/eval_gallary/resize_data1219/data_1219_dorm_15"
DATASET = 'data1219'


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 2 ** 16 / 1000.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim/denom

print(DATASET)

if DATASET == 'data0929':
    raw_depth_names = os.listdir(os.path.join(PATH, 'unscale_raw'))
    raw_depth_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
    raw_depth_paths = [os.path.join(PATH, 'unscale_raw', depth_name) for depth_name in raw_depth_names]
    
elif DATASET == 'data1219':
    raw_depth_names = os.listdir(os.path.join(PATH, 'unscale_raw'))
    raw_depth_names.sort(key=lambda x: int(x.split("_")[5][1:x.split("_")[5].find('.')]))
    raw_depth_paths = [os.path.join(PATH, 'unscale_raw', depth_name) for depth_name in raw_depth_names]

if DATASET == 'data0929':
    pred_depth_names = os.listdir(os.path.join(PATH, 'unscale_pred'))
    pred_depth_names.sort(key=lambda x: int(x[x.find("-")+1: x.find(".")]))
    pred_depth_paths = [os.path.join(PATH, 'unscale_pred', depth_name) for depth_name in pred_depth_names]
    
elif DATASET == 'data1219':
    pred_depth_names = os.listdir(os.path.join(PATH, 'unscale_pred'))
    pred_depth_names.sort(key=lambda x: int(x.split("_")[5][1:x.split("_")[5].find('.')]))
    pred_depth_paths = [os.path.join(PATH, 'unscale_pred', depth_name) for depth_name in pred_depth_names]


ssims = np.zeros(len(pred_depth_names))
mask_ssims = np.zeros(len(pred_depth_names))
psnrs = np.zeros(len(pred_depth_names))
mask_psnrs = np.zeros(len(pred_depth_names))
rmses = np.zeros(len(pred_depth_names))
mask_rmses = np.zeros(len(pred_depth_names))
maes = np.zeros(len(pred_depth_names))
mask_maes = np.zeros(len(pred_depth_names))
mses = np.zeros(len(pred_depth_names))
mask_mses = np.zeros(len(pred_depth_names))
t_valid = 0.0001

for i in range(len(raw_depth_names)):
    
    raw_name = raw_depth_names[i]
    pred_name = pred_depth_names[i]
    
    raw_path = raw_depth_paths[i]
    pred_path = pred_depth_paths[i]
    
    pred = Image.open(pred_path)    
    raw = Image.open(raw_path)
    
    pred = np.array(pred, dtype=np.uint16)    
    raw = np.array(raw, dtype=np.uint16)    

    if DATASET == 'data1219':
        pred = pred * 7.5 / 2**16
        raw = raw * 7.5 / 2**16
    else:
        pred = pred / 1000.0
        raw = raw / 1000.0
    
    mask = raw > t_valid
    num_valid = mask.sum()
    num_pixel = pred.shape[0] * pred.shape[1]
    
    mask_pred = pred[mask]
    mask_raw = raw[mask]
    
    diff = pred - raw
    diff_abs = np.abs(diff)
    diff_sqr = np.square(diff)
    
    mask_diff = mask_pred - mask_raw
    mask_diff_abs = np.abs(mask_diff)
    mask_diff_sqr = np.square(mask_diff)


    rmse = diff_sqr.sum() / num_pixel
    mses[i] = rmse
    rmse = np.sqrt(rmse)
    rmses[i] = rmse
    
    mask_rmse = mask_diff_sqr.sum() / (num_valid + 1e-8)
    mask_mses[i] = mask_rmse
    mask_rmse = np.sqrt(mask_rmse)
    mask_rmses[i] = mask_rmse

    mae = diff_abs.sum() / num_pixel
    maes[i] = mae
    mask_mae = mask_diff_abs.sum() / (num_valid + 1e-8)
    mask_maes[i] = mask_mae
    
    ssims[i] = ssim(pred, raw)
    mask_ssims[i] = ssim(mask_pred, mask_raw)
    psnrs[i] = peak_signal_noise_ratio(raw, pred, data_range= 2**16 / 1000.0)
    mask_psnrs[i] = peak_signal_noise_ratio(mask_raw, mask_pred, data_range= 2**16 /1000.0)
    
    # print('{} ssim'.format(pred_name), ssims[i])
    # print('{} psnr'.format(pred_name), psnrs[i])
    # print('{} mse'.format(pred_name), rmses[i])
    
        
    
print('ssim', np.mean(ssims))
print('mask ssim', np.mean(mask_ssims))

print('psnr', np.mean(psnrs))
print('mask psnr', np.mean(mask_psnrs))

print('rmse', np.mean(rmses))
print('mask rmse', np.mean(mask_rmses))

print('mae', np.mean(maes))
print('mask mae', np.mean(mask_maes))

print('mse', np.mean(mses))
print('mask mse', np.mean(mask_mses))
