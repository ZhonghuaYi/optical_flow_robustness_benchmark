import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda


def setup(seed: int, cudnn_enabled: bool, allow_tf32: bool, num_threads: int=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if cudnn_enabled:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    
    if allow_tf32:
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
    
    if num_threads is not None:
        torch.set_num_threads(num_threads)
