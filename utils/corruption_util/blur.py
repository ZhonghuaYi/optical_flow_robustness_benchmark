from typing import Tuple
import numpy as np
from PIL import Image
import os
import torch

from skimage.filters import gaussian
from numba import njit
from ..imagecorruptions import corrupt
from .util import check_image, _motion_blur
from .pth2img import super_fast_patch_wise_conv
from tqdm import tqdm


class GaussianBlur:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'GaussianBlur@{severity}'

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='gaussian_blur')
        image2 = corrupt(image2, severity=self.severity, corruption_name='gaussian_blur')
            
        return image1, image2
    

class DefocusBlur:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'DefocusBlur@{severity}'

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = corrupt(image1, severity=self.severity, corruption_name='defocus_blur')
        image2 = corrupt(image2, severity=self.severity, corruption_name='defocus_blur')
            
        return image1, image2


# Numba nopython compilation to shuffle_pixles
@njit()
def _shuffle_pixels_njit_glass_blur(d0, d1, x1, x2, c):

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(d0 - c[1], c[1], -1):
            for w in range(d1 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x1[h, w], x1[h_prime, w_prime] = x1[h_prime, w_prime], x1[h, w]
                x2[h, w], x2[h_prime, w_prime] = x2[h_prime, w_prime], x2[h, w]
    return x1, x2


class GlassBlur:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'GlassBlur@{severity}'
        
    def glass_blur(self, x1, x2, severity):
        assert x1.shape == x2.shape
        
        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][
            severity - 1]

        x1 = np.uint8(
            gaussian(np.array(x1) / 255., sigma=c[0], channel_axis=-1) * 255)
        x2 = np.uint8(
            gaussian(np.array(x2) / 255., sigma=c[0], channel_axis=-1) * 255)

        x1, x2 = _shuffle_pixels_njit_glass_blur(np.array(x1).shape[0], np.array(x1).shape[1], x1, x2, c)
        
        x1 = np.clip(gaussian(x1 / 255., sigma=c[0], channel_axis=-1), 0,
                    1) * 255
        x2 = np.clip(gaussian(x2 / 255., sigma=c[0], channel_axis=-1), 0,
                    1) * 255

        return x1, x2

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.glass_blur(image1, image2, self.severity)
            
        return image1, image2
    

class CameraMotionBlur:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'CameraMotionBlur@{severity}'
        
    def motion_blur(self, x1, x2, severity):
        assert x1.shape == x2.shape
        
        x1 = Image.fromarray(x1)
        x2 = Image.fromarray(x2)
        
        shape = np.array(x1).shape
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        x1 = np.array(x1)
        x2 = np.array(x2)

        angle = np.random.uniform(-45, 45)
        x1 = _motion_blur(x1, radius=c[0], sigma=c[1], angle=angle)
        x2 = _motion_blur(x2, radius=c[0], sigma=c[1], angle=angle)

        if len(x1.shape) < 3 or x1.shape[2] < 3:
            gray1 = np.clip(np.array(x1).transpose((0, 1)), 0, 255)
            gray2 = np.clip(np.array(x2).transpose((0, 1)), 0, 255)
            if len(shape) >= 3 or shape[2] >=3:
                return np.stack([gray1, gray1, gray1], axis=2), np.stack([gray2, gray2, gray2], axis=2)
            else:
                return gray1, gray2
        else:
            return np.clip(x1, 0, 255), np.clip(x2, 0, 255)

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1, image2 = self.motion_blur(image1, image2, self.severity)
            
        return image1, image2
    

class ObjectMotionBlur:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'CameraMotionBlur@{severity}'
        
    def object_motion_blur(self, image_sequence, keyframe_indices, severity):
        c = [3, 6, 9, 12, 15][severity - 1]
        
        blured_image_list = []
        
        print(f'Applying object motion blur on keyframes with severity {severity}')
        pbar = tqdm(range(len(keyframe_indices)))
        for i in pbar:
            keyframe_index = keyframe_indices[i]
            # 2c+1 frames
            image_stack = image_sequence[keyframe_index-c:keyframe_index+c+1, ...]
            
            blured_image = np.mean(image_stack, axis=0)
            blured_image_list.append(blured_image)
        
        return blured_image_list
    
    def __call__(self, image_sequence, keyframe_indices):
        blured_image_list = self.object_motion_blur(image_sequence, keyframe_indices, self.severity)
        return blured_image_list
    
    
class PSFBlur:
    def __init__(self, severity=5):
        self.severity = severity
        self.name = f'PSFBlur@{severity}'
    
    def psf_blur(self, image, severity):
        img_size = image.shape[:2]  # H, W
        
        patch_size = 16
        h_num = img_size[0] // patch_size
        w_num = img_size[1] // patch_size
        
        device = 'cuda'
        base_path = os.path.dirname(os.path.abspath(__file__))
        psf_path_list = [
            'psf/piece_6_fov20.0_fnum2.6_aper0_rms0.0296_idx8.pth',
            'psf/piece_1_fov34.0_fnum3.8_aper0_rms0.0832_idx2.pth',
            'psf/piece_6_fov28.0_fnum2.0_aper12_rms0.1102_idx8.pth',
            'psf/piece_3_fov22.0_fnum2.0_aper4_rms0.1588_idx5.pth',
            'psf/piece_4_fov20.0_fnum2.0_aper8_rms0.1939_idx4.pth'
        ]
        psf_path = os.path.join(base_path, psf_path_list[severity-1])
        psf = torch.load(psf_path, map_location=device)
        fov_pos = torch.linspace(0, 1, len(psf)).detach().cpu().numpy()
        
        torch_image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
        N, C, H, W = torch_image.shape
        pad_h1 = (img_size[0] - H) // 2
        pad_h2 = img_size[0] - H - pad_h1
        pad_w1 = (img_size[1] - W) // 2
        pad_w2 = img_size[1] - W - pad_w1
        
        img_pad = torch.nn.functional.pad(torch_image, (pad_w1, pad_w2, pad_h1, pad_h2), mode='reflect').to(device)
        
        blur_img = super_fast_patch_wise_conv(psf, img_pad, img_size=img_size, h_num=h_num, w_num=w_num, patch_length=patch_size, device=device, views_pos=fov_pos, fast=True)
        
        blur_img = blur_img[:, :, pad_h1:img_size[0]-pad_h2, pad_w1:img_size[1]-pad_w2]
        
        blur_img = blur_img[0].permute(1, 2, 0).cpu().numpy()
        blur_img = np.clip(blur_img * 255, 0, 255).astype(np.uint8)
        
        return blur_img
    
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check image
        image1 = check_image(image1)
        image2 = check_image(image2)
        # apply corruption
        image1 = self.psf_blur(image1, self.severity)
        image2 = self.psf_blur(image2, self.severity)
            
        return image1, image2
