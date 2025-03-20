import os.path as osp
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .flow_common import FlowDataset


class KITTI_C(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI-C', corruption='clean', severity=1):
        super(KITTI_C, self).__init__(aug_params, sparse=True)
        
        if corruption == 'clean':
            root = osp.join(root, 'clean', split)
        else:
            root = osp.join(root, f'{corruption}_{severity}', split)
        images1 = sorted(glob(osp.join(root, 'image_2', '*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2', '*_11.png')))
        
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

            self.flow_list = sorted(glob(osp.join(root, 'flow_occ', '*_10.png')))


