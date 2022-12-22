import os
import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import math

from PIL import Image
PATH = "gallary/eval_gallary/cost_spn_epoch2"
DRANGE = 2 ** 16 / 4000.0

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
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

def rmse(src, ref, size):
    diff_sqr = np.square(src - ref)
    return np.sqrt(diff_sqr.sum() / (size + 1e-8))
    
def mse(src, ref, size):
    diff_sqr = np.square(src - ref)
    return diff_sqr.sum() / (size + 1e-8)
    
def mae(src, ref, size):
    diff_abs = np.abs(src-ref)
    return diff_abs.sum() / (size + 1e-8)

def evaluate(src, ref, size):
    
    metrics = np.zeros(5)
    
    metrics[0] = rmse(src, ref, size)
    metrics[1] = mae(src, ref, size)
    metrics[2] = mse(src, ref, size)
    metrics[3] = ssim(src, ref)
    metrics[4] = peak_signal_noise_ratio(src, ref, data_range = DRANGE)
    
    return metrics

gt_depth_names = os.listdir(os.path.join(PATH, 'unscale_gt'))
gt_depth_names.sort(key=lambda x: x.split("_")[3]+x.split("_")[5])
gt_depth_paths = [os.path.join(PATH, 'unscale_gt', depth_name) for depth_name in gt_depth_names]
    

pred_depth_names = os.listdir(os.path.join(PATH, 'unscale_pred'))
pred_depth_names.sort(key=lambda x: x.split("_")[3]+x.split("_")[5])
pred_depth_paths = [os.path.join(PATH, 'unscale_pred', depth_name) for depth_name in pred_depth_names]

raw_depth_names = os.listdir(os.path.join(PATH, 'unscale_raw'))
raw_depth_names.sort(key=lambda x: x.split("_")[3]+x.split("_")[5])
raw_depth_paths = [os.path.join(PATH, 'unscale_raw', depth_name) for depth_name in raw_depth_names]

t_valid = 0.0001

mask_metrics = np.zeros((len(raw_depth_names), 5))
metrics = np.zeros((len(raw_depth_names), 5))
gt_metrics = np.zeros((len(raw_depth_names), 5))
gt_mask_metrics = np.zeros((len(raw_depth_names), 5))
mask_ssims = np.zeros(len(pred_depth_names))


for i in range(len(raw_depth_names)):
    
    raw_name = raw_depth_names[i]
    pred_name = pred_depth_names[i]
    gt_name = gt_depth_names[i]
    
    raw_path = raw_depth_paths[i]
    pred_path = pred_depth_paths[i]
    gt_path = gt_depth_paths[i]
    
    pred = Image.open(pred_path)    
    raw = Image.open(raw_path)
    gt = Image.open(gt_path)
    
    pred = np.array(pred, dtype=np.uint16)    
    raw = np.array(raw, dtype=np.uint16)    
    gt = np.array(gt, dtype=np.uint16)
    
    pred = pred / 4000.0
    raw = raw / 4000.0
    gt = gt / 4000.0
    
    mask = raw > t_valid
    num_valid = mask.sum()
    num_pixel = pred.shape[0] * pred.shape[1]
    
    mask_pred = pred[mask]
    mask_raw = raw[mask]
    mask_gt = gt[mask]
    
    mask_ssims[i] = ssim(mask_pred, mask_raw)

    metrics[i] = evaluate(raw, pred, num_pixel)
    mask_metrics[i] = evaluate(mask_pred, mask_raw, num_valid)
    gt_metrics[i] = evaluate(gt, pred, num_pixel)
    gt_mask_metrics[i] = evaluate(mask_gt ,mask_pred, num_valid)
    
print(np.mean(metrics, axis=0))
print(np.mean(mask_metrics, axis=0))
print(np.mean(gt_metrics, axis=0))
print(np.mean(gt_mask_metrics, axis=0))


        
