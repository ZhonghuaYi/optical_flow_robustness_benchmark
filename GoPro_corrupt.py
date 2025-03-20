import os
from os import PathLike
from glob import glob
import numpy as np
import cv2 as cv
from tqdm import tqdm
from rich import print
import subprocess

from utils.image_corruptions import get_corruption


SEQUENCE_LIST = [
    'GOPR0374_11_02',
    'GOPR0379_11_00',
    'GOPR0384_11_02',
    'GOPR0385_11_00',
    'GOPR0386_11_00',
]


def save_clean_image_pairs(interpolated_root: str, clean_root: str):
    """
    Save clean image pairs from interpolated images.
    
    Args:
        interpolated_root: str, path to the interpolated images.
        clean_root: str, path to save the clean image pairs.
        scale: int, scale factor of the interpolated
        
    """
    if not os.path.exists(interpolated_root):
        raise FileNotFoundError(f'Interpolated root not found: {interpolated_root}')
    
    for sequence in SEQUENCE_LIST:
        # load interpolated images
        interpolated_sequence = os.path.join(interpolated_root, f'{sequence}_4x_converted')
        frames = sorted(glob(os.path.join(interpolated_sequence, '*.png')))
        
        print(f'Interpolated sequence: [yellow]{interpolated_sequence}[/yellow]: {len(frames)} frames')
        
        frames_indices = list(range(len(frames)))
        
        # # get keyframes, original frame rate is 240fps, we sample 30fps
        # keyframes_indices = frames_indices[::4*8]
        
        ### 10fps
        keyframes_indices = frames_indices[::4*8*3]
        
        
        # drop the first and last keyframes
        keyframes_indices = keyframes_indices[1:-1]
        print(f'{len(keyframes_indices)} keyframes')
        
        np.save(os.path.join(interpolated_sequence, 'keyframes_indices.npy'), np.array(keyframes_indices))
        
        clean_sequence_path = os.path.join(clean_root, sequence)
        if not os.path.exists(clean_sequence_path):
            os.makedirs(clean_sequence_path)
        
        count = 0
        pbar = tqdm(range(len(keyframes_indices)-1))
        for i in pbar:
            keyframe_index = keyframes_indices[i]
            next_keyframe_index = keyframes_indices[i+1]
            frame0 = cv.imread(frames[keyframe_index])
            frame1 = cv.imread(frames[next_keyframe_index])
            frame0_name = os.path.join(clean_sequence_path, f'{keyframe_index:06d}_0.png')
            frame1_name = os.path.join(clean_sequence_path, f'{keyframe_index:06d}_1.png')
            cv.imwrite(frame0_name, frame0)
            cv.imwrite(frame1_name, frame1)
            count += 1
            
        print(f'Origin: {len(frames)} frames, Keyframes: {len(keyframes_indices)} frames, Saved: {count} pairs')


def load_clean_image_pairs(clean_root: str, sequence: str):
    """
    Load clean image pairs from a sequence.
    
    Args:
        clean_root: str, path to the clean images.
        sequence: str, sequence name.
        
    Returns:
        list, list of clean image pairs.
    """
    sequence_path = os.path.join(clean_root, sequence)
    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f'Clean sequence not found: {sequence_path}')
    
    clean_image_pairs = []
    frames0 = sorted(glob(os.path.join(sequence_path, '*_0.png')))
    frames1 = sorted(glob(os.path.join(sequence_path, '*_1.png')))
    for frame0, frame1 in zip(frames0, frames1):
        frames0_name = os.path.basename(frame0)
        frames1_name = os.path.basename(frame1)
        assert frames0_name[:-6] == frames1_name[:-6]
        clean_image_pairs.append([frame0, frame1])
    
    return clean_image_pairs


def corrupt_image_pairs_of_sequence(clean_image_pairs, corrupt_sequence_root, corruption_name: str, severity: int):
    """
    Corrupt clean image pairs.
    
    Args:
        clean_image_pairs: list, list of clean image pairs.
        corruption_name: str, name of the corruption.
        severity: int, severity of the corruption.
        
    Returns:
        list, list of corrupted image pairs.
    """
    pbar = tqdm(range(len(clean_image_pairs)))
    for i in pbar:
        frame0, frame1 = clean_image_pairs[i]
        frame0_name = os.path.basename(frame0)
        frame1_name = os.path.basename(frame1)
        
        frame0 = cv.imread(frame0)
        frame1 = cv.imread(frame1)
        
        corruption = get_corruption(corruption_name, severity)
        corrupted_frame0, corrupted_frame1 = corruption(frame0, frame1)
        corrupted_frame0_name = os.path.join(corrupt_sequence_root, frame0_name)
        corrupted_frame1_name = os.path.join(corrupt_sequence_root, frame1_name)
        
        cv.imwrite(corrupted_frame0_name, corrupted_frame0)
        cv.imwrite(corrupted_frame1_name, corrupted_frame1)
        
        
def corrupt_images_pairs(clean_root: str, corrupted_root: str, corruption_name: str, severity: int):
    """
    Corrupt clean image pairs.
    
    Args:
        clean_root: str, path to the clean images.
        corrupted_root: str, path to save the corrupted images.
        corruption_name: str, name of the corruption.
        severity: int, severity of the corruption.
    """
    for sequence in SEQUENCE_LIST:
        print(f'{sequence}')
        clean_image_pairs = load_clean_image_pairs(clean_root, sequence)
        corrupt_sequence_root = os.path.join(corrupted_root, sequence)
        if not os.path.exists(corrupt_sequence_root):
            os.makedirs(corrupt_sequence_root)
        print(f'Corrupted sequence dir: [yellow]{corrupt_sequence_root}[/yellow]')
        
        corrupt_image_pairs_of_sequence(clean_image_pairs, corrupt_sequence_root, corruption_name, severity)
        

def corrupt_object_motion_from_interpolated_sequence(interpolated_root, corrupted_root, severity, corruption_name='object_motion_blur'):
    if not os.path.exists(interpolated_root):
        raise FileNotFoundError(f'Interpolated root not found: {interpolated_root}')
    
    corruption = get_corruption(corruption_name, severity)
    
    for sequence in SEQUENCE_LIST:
        print(f'Start corrupting {sequence}')
        # load interpolated images
        interpolated_sequence = os.path.join(interpolated_root, f'{sequence}_4x_converted')
        frames = sorted(glob(os.path.join(interpolated_sequence, '*.png')))
        
        print(f'Interpolated sequence: [yellow]{interpolated_sequence}[/yellow]: {len(frames)} frames')
        
        # load all frames into numpy array
        frames_list = []
        print(f'Loading frames from {interpolated_sequence}')
        pbar = tqdm(range(len(frames)))
        for i in pbar:
            frame = cv.imread(frames[i])
            frames_list.append(frame)
        frames_list = np.array(frames_list)
        
        # load keyframes indices
        keyframes_indices = np.load(os.path.join(interpolated_sequence, 'keyframes_indices.npy'))
        print(f'{len(keyframes_indices)} keyframes')
        
        # blur keyframes
        blured_keyframe_list = corruption(frames_list, keyframes_indices)
        print(f'Blured image list: {len(blured_keyframe_list)}')
        
        # save blured images into corrupted root in form of image pair
        corrupted_sequence_path = os.path.join(corrupted_root, sequence)
        if not os.path.exists(corrupted_sequence_path):
            os.makedirs(corrupted_sequence_path)
        
        count = 0
        pbar = tqdm(range(len(blured_keyframe_list)-1))
        for i in pbar:
            blured_keyframe_0 = blured_keyframe_list[i]
            blured_keyframe_1 = blured_keyframe_list[i+1]
            blured_keyframe_0_name = os.path.join(corrupted_sequence_path, f'{keyframes_indices[i]:06d}_0.png')
            blured_keyframe_1_name = os.path.join(corrupted_sequence_path, f'{keyframes_indices[i]:06d}_1.png')
            cv.imwrite(blured_keyframe_0_name, blured_keyframe_0)
            cv.imwrite(blured_keyframe_1_name, blured_keyframe_1)
            count += 1
            
        print(f'Origin: {len(frames)} frames, Keyframes: {len(keyframes_indices)} frames, Saved: {count} pairs')
        
        

def interpolated_sequence_to_mp4(interpolated_sequence_root, fps):
    """
    Convert image sequence to mp4 video.
    
    Args:
        image_sequence_root: str, path to the image sequence.
        mp4_path: str, path to save the mp4 video.
    """
    
    for sequence in SEQUENCE_LIST:
        interpolated_sequence = os.path.join(interpolated_sequence_root, f'{sequence}_4x')
        images = os.path.join(interpolated_sequence, '%06d.png')
        mp4_path = os.path.join(interpolated_sequence_root, f'{sequence}.mp4')
        return_code = subprocess.call(["/usr/bin/ffmpeg", "-y", "-framerate", str(fps), "-i", images, "-c:v", "libx264", "-preset", "veryslow", "-crf", "0", "-pix_fmt", "yuv420p", mp4_path])
        print(f'Convert {interpolated_sequence} to {mp4_path}: {return_code}')
    
    
def mp4_to_interpolated_sequence(mp4_root, interpolated_sequence_root):
    """
    Convert mp4 video to image sequence.
    
    Args:
        mp4_path: str, path to the mp4 video.
        interpolated_sequence_root: str, path to save the image sequence.
    """
    for sequence in SEQUENCE_LIST:
        mp4_path = os.path.join(mp4_root, f'{sequence}.mp4')
        interpolated_sequence = os.path.join(interpolated_sequence_root, f'{sequence}_4x_converted')
        if not os.path.exists(interpolated_sequence):
            os.makedirs(interpolated_sequence)
        return_code = subprocess.call(["/usr/bin/ffmpeg", "-i", mp4_path, os.path.join(interpolated_sequence, "%06d.png")])
        print(f'Convert {mp4_path} to {interpolated_sequence}: {return_code}')

        imgs = sorted(glob(os.path.join(interpolated_sequence, '*.png')))
        print(f'fix image names in {interpolated_sequence}')
        for i in range(len(imgs)):
            os.rename(imgs[i], os.path.join(interpolated_sequence, f'{i:06d}.png'))
            

def corruption_video(corruption_name: str, severity: int, clean_video_root: str, corrupted_root: str, ffmpeg_root='/usr/bin'):
    """
    Corrupt video.
    
    Args:
        corruption_name: str, name of the corruption.
        severity: int, severity of the corruption.
        video_path: str, path to the video.
        corrupted_video_path: str, path to save the corrupted video.
    """
    for sequence in SEQUENCE_LIST:
        print(f'{sequence}')
        
        ### video corruption
        clean_video_path = os.path.join(clean_video_root, f'{sequence}.mp4')  # GoPro-C/4x/*.mp4
        corrupted_video_path = os.path.join(corrupted_root, f'{sequence}.mp4')  # GoPro-C/{corruption_name}_{severity}/*.mp4
        print(f'[yellow]Corrupted video path: {corrupted_video_path}')
        if not os.path.exists(os.path.dirname(corrupted_video_path)):
            os.makedirs(os.path.dirname(corrupted_video_path))
        corruption = get_corruption(corruption_name, severity)
        corruption(clean_video_path, corrupted_video_path, ffmpeg_root)
        
        ### convert corrupted video to image sequence
        corrupted_sequence_converted_root = os.path.join(corrupted_root, f'{sequence}_4x_converted')  # GoPro-C/{corruption_name}_{severity}/*_4x_converted
        if not os.path.exists(corrupted_sequence_converted_root):
            os.makedirs(corrupted_sequence_converted_root)
        return_code = subprocess.call(["/usr/bin/ffmpeg", "-i", corrupted_video_path, os.path.join(corrupted_sequence_converted_root, "%06d.png")])
        print(f'[yellow]Convert {corrupted_video_path} to {corrupted_sequence_converted_root}: {return_code}')
        imgs = sorted(glob(os.path.join(corrupted_sequence_converted_root, '*.png')))
        print(f'fix image names in {corrupted_sequence_converted_root}')
        for i in range(len(imgs)):
            os.rename(imgs[i], os.path.join(corrupted_sequence_converted_root, f'{i:06d}.png'))
            
            
        ### converted sequence to image pairs
        keyframes_indices = np.load(os.path.join(clean_video_root, f'{sequence}_4x_converted', 'keyframes_indices.npy'))
        corrupted_pairs_path = os.path.join(corrupted_root, sequence)
        if not os.path.exists(corrupted_pairs_path):
            os.makedirs(corrupted_pairs_path)
        count = 0
        pbar = tqdm(range(len(keyframes_indices)-1))
        for i in pbar:
            keyframe_index = keyframes_indices[i]
            next_keyframe_index = keyframes_indices[i+1]
            frame0 = cv.imread(os.path.join(corrupted_sequence_converted_root, f'{keyframe_index:06d}.png'))
            frame1 = cv.imread(os.path.join(corrupted_sequence_converted_root, f'{next_keyframe_index:06d}.png'))
            frame0_name = os.path.join(corrupted_pairs_path, f'{keyframe_index:06d}_0.png')
            frame1_name = os.path.join(corrupted_pairs_path, f'{keyframe_index:06d}_1.png')
            cv.imwrite(frame0_name, frame0)
            cv.imwrite(frame1_name, frame1)
            count += 1
            
        print(f'[yellow]Keyframes: {len(keyframes_indices)} frames, Saved: {count} pairs')
        


if __name__ == '__main__':
    ## convert interpolated sequence to mp4
    interpolated_root = f'data/GoPro-C-10/4x'
    fps = 960
    interpolated_sequence_to_mp4(interpolated_root, fps)
    mp4_to_interpolated_sequence(interpolated_root, interpolated_root)
    
    ## save clean image pairs
    clean_root = f'data/GoPro-C-10/clean'
    save_clean_image_pairs(interpolated_root, clean_root)
    
    ## convert mp4 to sequence
    
    ######
    # corrupt image pairs
    ######
    
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
            corrupted_root = f'data/GoPro-C-10/{corruption_name}_{s}'
            print(f'{corruption_name}-{s}')
            corrupt_images_pairs(clean_root, corrupted_root, corruption_name, s)
    
    ########
    # corrupt object motion
    ########
    
    severity = [1, 2, 3, 4, 5]
    for s in severity:
        corrupted_root = f'data/GoPro-C-10/object_motion_blur_{s}'
        print(f'object_motion_blur-{s}')
        corrupt_object_motion_from_interpolated_sequence(interpolated_root, corrupted_root, s)
    
    
    # #######
    # video corruption
    # #######
    
    severity = [1, 2, 3, 4, 5]
    corruption_name_list = [
        'h264_crf',
        'h264_abr',
        'bit_error'
    ]
    ffmpeg_root = '/mnt/sto/yzh/workspace/FFmpeg'
    clean_video_root = f'data/GoPro-C-10/4x'
    for corruption_name in corruption_name_list:
        for s in severity:
            # s = 5
            corrupted_root = f'data/GoPro-C-10/{corruption_name}_{s}'
            print(f'[yellow]{corruption_name}-{s}')
            corruption_video(corruption_name, s, clean_video_root, corrupted_root)
            
        #     break
        # break
    
    
