import sys, os
import random
import numpy as np
from PIL import Image
from skimage.transform import resize as numpy_resize
import warnings

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# Reference: https://github.com/tsunghan-wu/Depth-Completion/blob/master/depth_completion/data/data_loader.py

class Data0929(Dataset):
    
    
    def __init__(self, data_path, train=True):

        self.data_root = data_path
        self.len = 0
        self.train = train
        
        self._load_data_name()
   
    def _load_data_name(self):
        self.color_name = os.listdir(os.path.join(self.data_root, "cropped_rgb"))
        self.depth_name = os.listdir(os.path.join(self.data_root, "cropped_interpo_depth"))
        self.scene_name = [name[:name.find(".")] for name in self.color_name]
        self.len = len(self.color_name)
        

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        depth = np.array(Image.open(self.depth_name[index]))
        color = np.array(Image.open(self.color_name[index]))
        
        batch = {'scene_name' : self.scene_name[index], 'color' : color, 'depth' : depth}
        
        return batch

def customed_collate_fn(dataset_name):
    if dataset_name == 'data0929':
        return customed_collate_fn_data0929
    else:
        raise Exception('Not recognized dataset name')

def customed_collate_fn_data0929(batch):
    # trans_height, trans_width = 512, 640
    trans_height, trans_width = 864, 1080
    tensor_transform = transforms.Compose([
        transforms.Resize((trans_height, trans_width)),
        transforms.ToTensor(),
    ])
    def numpy_transform(value):
        if value.shape[0] != trans_height or value.shape[1] != trans_width:
            value = numpy_resize(value, (trans_height, trans_width), mode='constant', anti_aliasing=False)
        value = torch.tensor(value).type(torch.float32)
        
        return value
    def _transform_fn(key, value):
        if key == 'depth':
            value = numpy_transform(value)
            value /= 1000.00
            value = torch.unsqueeze(value, 0)
        elif key == 'color':
            value = numpy_transform(value)
            value /= 255
            value = value.permute(2, 0, 1)
        return value
            
    keys = list(batch[0].keys())
    values = {}
    for key in keys:
        if key == 'scene_name':
            values[key] = [one_batch[key] for one_batch in batch]
        else:
            arr = [_transform_fn(key, one_batch[key]) for one_batch in batch]
            this_value = torch.stack(arr, 0, out=None)
            values[key] = this_value
    return values