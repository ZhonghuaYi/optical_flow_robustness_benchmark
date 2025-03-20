import os.path as osp
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .flow_common import FlowDataset
from .dataset_util import frame_utils


class GoPro_C:
    def __init__(self, root='datasets/GoPro-C', corruption='clean', severity=1):
        
        self.SEQUENCE_LIST = [
            'GOPR0374_11_02',
            'GOPR0379_11_00',
            'GOPR0384_11_02',
            'GOPR0385_11_00',
            'GOPR0386_11_00',
        ]
        
        self.image_list = []
        self.extra_info = []
        
        if corruption == 'clean':
            root = osp.join(root, 'clean')
        else:
            root = osp.join(root, f'{corruption}_{severity}')
            
        for seq in self.SEQUENCE_LIST:
            images1 = sorted(glob(osp.join(root, seq, '*_0.png')))
            images2 = sorted(glob(osp.join(root, seq, '*_1.png')))
            
            for img1, img2 in zip(images1, images2):
                frame_id = img1.split('/')[-1]
                self.extra_info += [ [seq+frame_id] ]
                self.image_list += [ [img1, img2] ]
                
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        index = index % len(self.image_list)
        
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        
        return img1, img2, None, None
        
        
    