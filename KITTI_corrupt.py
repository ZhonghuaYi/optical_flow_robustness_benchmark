import os
from os import PathLike
from glob import glob
import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils.image_corruptions import get_corruption


def load_clean_KITTI(clean_root: PathLike):
    training_root = os.path.join(clean_root, 'training')
    val_root = os.path.join(clean_root, 'val')
    
    # Load training images and flow
    training_image_list = []
    training_images1 = sorted(glob(os.path.join(training_root, 'image_2', '*_10.png')))
    training_images2 = sorted(glob(os.path.join(training_root, 'image_2', '*_11.png')))
    training_flow_list = sorted(glob(os.path.join(training_root, 'flow_occ', '*_10.png')))
    for img1, img2 in zip(training_images1, training_images2):
        training_image_list += [[img1, img2]]
    
    # Load validation images and flow
    val_image_list = []
    val_images1 = sorted(glob(os.path.join(val_root, 'image_2', '*_10.png')))
    val_images2 = sorted(glob(os.path.join(val_root, 'image_2', '*_11.png')))
    val_flow_list = sorted(glob(os.path.join(val_root, 'flow_occ', '*_10.png')))
    for img1, img2 in zip(val_images1, val_images2):
        val_image_list += [[img1, img2]]
    
    return training_image_list, training_flow_list, val_image_list, val_flow_list


def corrupt_kitti(clean_root, corrupted_root, corruption_name, severity):
    training_image_list, training_flow_list, val_image_list, val_flow_list = load_clean_KITTI(clean_root)
    
    pbar = tqdm(range(len(training_image_list)))
    # Corrupt training images
    for i in pbar:
        image1 = training_image_list[i][0]
        image2 = training_image_list[i][1]
        flow = training_flow_list[i]
        
        # Save corrupted images
        image1_name = os.path.basename(image1)
        image2_name = os.path.basename(image2)
        flow_name = os.path.basename(flow)
        
        image1 = cv.imread(image1, cv.IMREAD_COLOR)
        image2 = cv.imread(image2, cv.IMREAD_COLOR)
        flow = cv.imread(flow, cv.IMREAD_ANYDEPTH|cv.IMREAD_COLOR)
        
        # Apply corruption
        corruption = get_corruption(corruption_name, severity)
        image1, image2 = corruption(image1, image2)
        
        image1_path = os.path.join(corrupted_root, 'training', 'image_2', image1_name)
        image2_path = os.path.join(corrupted_root, 'training', 'image_2', image2_name)
        flow_path = os.path.join(corrupted_root, 'training', 'flow_occ', flow_name)
        
        os.makedirs(os.path.dirname(image1_path), exist_ok=True)
        os.makedirs(os.path.dirname(image2_path), exist_ok=True)
        os.makedirs(os.path.dirname(flow_path), exist_ok=True)
        
        cv.imwrite(image1_path, image1)
        cv.imwrite(image2_path, image2)
        cv.imwrite(flow_path, flow)
        
    pbar = tqdm(range(len(val_image_list)))
    # Corrupt validation images
    for i in pbar:
        image1 = val_image_list[i][0]
        image2 = val_image_list[i][1]
        flow = val_flow_list[i]
        
        # Save corrupted images
        image1_name = os.path.basename(image1)
        image2_name = os.path.basename(image2)
        flow_name = os.path.basename(flow)
        
        image1 = cv.imread(image1, cv.IMREAD_COLOR)
        image2 = cv.imread(image2, cv.IMREAD_COLOR)
        flow = cv.imread(flow, cv.IMREAD_ANYDEPTH|cv.IMREAD_COLOR)
        
        # Apply corruption
        corruption = get_corruption(corruption_name, severity)
        image1, image2 = corruption(image1, image2)
        
        image1_path = os.path.join(corrupted_root, 'val', 'image_2', image1_name)
        image2_path = os.path.join(corrupted_root, 'val', 'image_2', image2_name)
        flow_path = os.path.join(corrupted_root, 'val', 'flow_occ', flow_name)
        
        os.makedirs(os.path.dirname(image1_path), exist_ok=True)
        os.makedirs(os.path.dirname(image2_path), exist_ok=True)
        os.makedirs(os.path.dirname(flow_path), exist_ok=True)
        
        cv.imwrite(image1_path, image1)
        cv.imwrite(image2_path, image2)
        cv.imwrite(flow_path, flow)
        

if __name__ == '__main__':
    clean_root = 'data/KITTI-C/clean'
    corruption_name_list = [
        'jpeg_compression', 
        'pixelate', 
        'contrast', 
        'brightness',
        'low_light',
        'saturate',
        'over_exposure',
        'under_exposure',
        'spatter', 
        'fog', 
        'frost', 
        'snow', 
        'gaussian_noise', 
        'shot_noise', 
        'impulse_noise', 
        'gaussian_blur', 
        'defocus_blur', 
        'glass_blur', 
        'camera_motion_blur',
        'psf_blur'
        ]

    severity = [1, 2, 3, 4, 5]
    
    for corruption_name in corruption_name_list:
        for s in severity:
            corrupted_root = f'data/KITTI-C/{corruption_name}_{s}'
            print(f'{corruption_name}-{s}')
            corrupt_kitti(clean_root, corrupted_root, corruption_name, s)

