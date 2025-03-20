import os
from glob import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def split():
    """
    Split KITTI training data into training and validation sets.
    The resulted dataset is the 'clean' version of KITTI-C.
    
    The KITTI training data contains 200 image pairs with ground truth flow.
    This function randomly selects 80 image pairs as validation set and moves them
    to a separate validation folder, leaving 120 pairs for training.
    
    The folder structure will be:
        data/KITTI/
            training/
                image_2/     # Contains 120 image pairs
                flow_occ/    # Contains 120 flow files
            val/
                image_2/     # Contains 80 image pairs 
                flow_occ/    # Contains 80 flow files
    """
    
    training_path = "data/KITTI/training"
    val_path = "data/KITTI/val"
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    images1_paths = sorted(glob(os.path.join(training_path, 'image_2', '*_10.png')))
    images2_paths = sorted(glob(os.path.join(training_path, 'image_2', '*_11.png')))
    flow_occ_paths = sorted(glob(os.path.join(training_path, 'flow_occ', '*_10.png')))


    assert len(images1_paths) == len(images2_paths) == len(flow_occ_paths)

    val_index = random.sample(range(len(images1_paths)), 80)

    for i in range(len(val_index)):
        image1 = images1_paths[val_index[i]]
        image2 = images2_paths[val_index[i]]
        flow_occ = flow_occ_paths[val_index[i]]
        
        image1_name = os.path.basename(image1)
        image2_name = os.path.basename(image2)
        flow_occ_name = os.path.basename(flow_occ)
        
        image1_val_path = os.path.join(val_path, 'image_2', image1_name)
        image2_val_path = os.path.join(val_path, 'image_2', image2_name)
        flow_occ_val_path = os.path.join(val_path, 'flow_occ', flow_occ_name)
        
        os.makedirs(os.path.dirname(image1_val_path), exist_ok=True)
        os.makedirs(os.path.dirname(flow_occ_val_path), exist_ok=True)
            
        os.rename(image1, image1_val_path)
        os.rename(image2, image2_val_path)
        os.rename(flow_occ, flow_occ_val_path)
        

def flow_amplitude_statistic():
    clean_flow_root = "data/KITTI-C/clean/training/flow_occ"
    val_flow_root = "data/KITTI-C/clean/val/flow_occ"
    
    clean_amp_list = []
    clean_percent_list = []
    val_amp_list = []
    val_percent_list = []
    
    th_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    clean_flow_paths = sorted(glob(os.path.join(clean_flow_root, '*.png')))
    val_flow_paths = sorted(glob(os.path.join(val_flow_root, '*.png')))
    
    for flow_file in clean_flow_paths:
        flow, valid = readFlowKITTI(flow_file)
        amp = np.linalg.norm(flow, axis=2)
        amp = amp.flatten()
        valid = valid.flatten()
        amp_valid = amp[valid > 0]
        clean_amp_list.append(amp_valid)
    
    clean_amp_list = np.concatenate(clean_amp_list)
    
    for i in range(len(th_list)+1):
        th_low = th_list[i-1] if i > 0 else 0
        th_high = th_list[i] if i < len(th_list) else np.inf
        percent = np.mean((clean_amp_list > th_low) & (clean_amp_list <= th_high))
        clean_percent_list.append(percent)
        
    for flow_file in val_flow_paths:
        flow, valid = readFlowKITTI(flow_file)
        amp = np.linalg.norm(flow, axis=2)
        amp = amp.flatten()
        valid = valid.flatten()
        amp_valid = amp[valid > 0]
        val_amp_list.append(amp_valid)
        
    val_amp_list = np.concatenate(val_amp_list)
    np.save("val_amp.npy", val_amp_list)
    
    for i in range(len(th_list)+1):
        th_low = th_list[i-1] if i > 0 else 0
        th_high = th_list[i] if i < len(th_list) else np.inf
        percent = np.mean((val_amp_list > th_low) & (val_amp_list <= th_high))
        val_percent_list.append(percent)
        
    print(f"clean flow amplitude statistic: {clean_percent_list}")
    print(f"val flow amplitude statistic: {val_percent_list}")
    
    # th_list = th_list.append(-1)
    th_list = th_list + [110]
    plt.plot(np.array(th_list), np.array(clean_percent_list), label='training')
    plt.plot(np.array(th_list), np.array(val_percent_list), label='val')
    plt.legend()
    plt.savefig("flow_amplitude_statistic.png")
    

if __name__ == "__main__":
    flow_amplitude_statistic()

