import sys
import logging
import os
import shutil
from tqdm import tqdm
import numpy as np
import cv2 as cv
import torch
from omegaconf import OmegaConf
from rich import print
import json
import time
import argparse

from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate

sys.path.append('../../')
from datasets.GoPro_C import GoPro_C
from datasets.dataset_util.flow_viz import flow_to_image
from metrics.metric import CRE, rCRE, EPE
from utils.common import setup


@torch.no_grad()
def benchmark_GoPro_C(data_root, split, corruption, severity, raft_model, raft_iters, save_flow_path=None):
    
    if corruption == 'clean':
        dataset_clean = GoPro_C(root=data_root, corruption='clean', severity=severity)
        dataset_c = None
    else:
        dataset_clean = GoPro_C(root=data_root, corruption='clean', severity=severity)
        dataset_c = GoPro_C(root=data_root, corruption=corruption, severity=severity)
        assert len(dataset_clean) == len(dataset_c), 'The number of samples in clean and corrupted datasets must be the same'
        
    print(f'Number of samples: {len(dataset_clean)}')
    
    results = {
        # 'CRE': [],
        'rCRE': [],
        # 'clean_EPE': [],
        # 'corrupt_EPE': [],
        # 'amp': []
    }
    
    pbar_clean = tqdm(range(len(dataset_clean)), total=len(dataset_clean))
    for i in pbar_clean:
        # load clean data
        image1_clean, image2_clean, _, _ = dataset_clean[i]
        image1_clean = image1_clean.unsqueeze(0).cuda()
        image2_clean = image2_clean.unsqueeze(0).cuda()
        # flow_gt = flow_gt.unsqueeze(0).cuda()
        # valid_gt = valid_gt.unsqueeze(0).cuda()
        
        # pad the image
        padder = InputPadder(image1_clean.shape)
        image1_clean, image2_clean = padder.pad(image1_clean, image2_clean)
        
        # model inference
        _, flow_clean = raft_model(image1_clean, image2_clean, iters=raft_iters, test_mode=True)
        
        # unpad the flow and the image
        flow_clean = padder.unpad(flow_clean)
        image1_clean = padder.unpad(image1_clean)
        image2_clean = padder.unpad(image2_clean)
        
        mask_clean = torch.ones_like(flow_clean[:, 0:1, :, :], device=flow_clean.device)
        
        # amplitude = flow_clean.pow(2).sum(dim=1).sqrt().view(-1)
        # masked_amplitude = amplitude[mask_clean.view(-1) > 0]
        # results['amp'].append(masked_amplitude.cpu().numpy())
        
        # # compute EPE of clean data
        # epe_clean = EPE(flow_clean, flow_gt, (valid_gt * mask_clean) > 0)
        # results['clean_EPE'].append(epe_clean)
        
        if dataset_c is not None:
            # load corrupted data
            image1_corrupt, image2_corrupt, _, _ = dataset_c[i]
            image1_corrupt = image1_corrupt.unsqueeze(0).cuda()
            image2_corrupt = image2_corrupt.unsqueeze(0).cuda()
            
            # pad the image
            image1_corrupt, image2_corrupt = padder.pad(image1_corrupt, image2_corrupt)
            
            # model inference
            _, flow_corrupt = raft_model(image1_corrupt, image2_corrupt, iters=raft_iters, test_mode=True)
            
            # unpad the flow and the image
            flow_corrupt = padder.unpad(flow_corrupt)
            image1_corrupt = padder.unpad(image1_corrupt)
            image2_corrupt = padder.unpad(image2_corrupt)
            
            mask_corrupt = torch.ones_like(flow_corrupt[:, 0:1, :, :], device=flow_corrupt.device)
            
            # # compute EPE of corrupted data
            # epe_corrupt = EPE(flow_corrupt, flow_gt, (valid_gt * mask_corrupt) > 0)
            # results['corrupt_EPE'].append(epe_corrupt)
            
            # mask_final = (valid_gt * mask_clean * mask_corrupt) > 0
            mask_final = (mask_clean * mask_corrupt) > 0
            # # compute CRE
            # cre = CRE(flow_clean, flow_corrupt, flow_gt, mask_final)
            # results['CRE'].append(cre)
            
            # compute rCRE
            rcre = rCRE(flow_clean, flow_corrupt, mask_final)
            results['rCRE'].append(rcre)
            
            if save_flow_path is not None:
                
                flow_np = flow_corrupt[0].permute(1, 2, 0).cpu().numpy()
                flow_image = flow_to_image(flow_np)
                img_name = dataset_clean.extra_info[i][0]
                img_path = os.path.join(save_flow_path, f'{img_name}')
                pbar_clean.set_description(f'Saving flow image {img_path}')
                cv.imwrite(img_path, flow_image)
        else:
            if save_flow_path is not None:
                
                flow_np = flow_clean[0].permute(1, 2, 0).cpu().numpy()
                flow_image = flow_to_image(flow_np)
                img_name = dataset_clean.extra_info[i][0]
                img_path = os.path.join(save_flow_path, f'{img_name}')
                pbar_clean.set_description(f'Saving flow image {img_path}')
                cv.imwrite(img_path, flow_image)
                
    for key in results.keys():
        if len(results[key]) > 0:
            results[key] = torch.cat(results[key]).mean().item()
        
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='RAFT_ood', help="name your experiment")
    args = parser.parse_args()
    
    # Load the configuration file
    data_root = '../../data/GoPro-C-10'
    split = 'val'
    corruption_name_list = [
        # # 'clean',
        # 'jpeg_compression', 
        # 'pixelate', 
        # # 'elastic_transform', 
        # 'contrast', 
        # 'saturate',
        # # 'line_dropout',
        # 'brightness',
        # 'low_light',
        # 'over_exposure',
        # 'under_exposure',
        # 'spatter', 
        # 'fog', 
        # 'frost', 
        # 'snow', 
        # 'gaussian_noise', 
        # 'shot_noise', 
        # 'impulse_noise', 
        # # 'speckle_noise', 
        # 'gaussian_blur', 
        # 'defocus_blur', 
        # 'glass_blur', 
        # 'camera_motion_blur',
        # 'object_motion_blur',
        # 'psf_blur',
        'h264_abr',
        'h264_crf',
        'bit_error'
        ]
    result_terms = [
        # 'CRE',
        'rCRE',
        # 'clean_EPE',
        # 'corrupt_EPE',
    ]
    all_result_dict = {}
    for k in corruption_name_list:
        all_result_dict[k] = {ik : [] for ik in result_terms}
        all_result_dict[k].update({f'{ik}_mean' : 0.0 for ik in result_terms})
        all_result_dict[k].update({f'result' : ''})
    
    severity_list = [1, 2, 3, 4, 5]
    save_flow_path = None
    
    # load configuration file
    RAFT_config = OmegaConf.load(f'./{args.config}.yaml')
    project_name = RAFT_config.project_name
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    project_name = f'{project_name}_{current_time}'
    # setup the environment
    setup(RAFT_config.seed, RAFT_config.cudnn_enabled, RAFT_config.allow_tf32, RAFT_config.num_threads)
    
    # Load the configuration file
    os.makedirs(f'./benchmark_results/{project_name}', exist_ok=True)
    log_path = f'./benchmark_results/{project_name}/results.log'
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w', format='%(message)s')
    
    # load the model
    raft_model = RAFT(RAFT_config)
    model_dict = torch.load(RAFT_config.model_path)
    for key in list(model_dict.keys()):
        model_dict[key.replace('module.', '')] = model_dict.pop(key)
    raft_model.load_state_dict(model_dict)
    
    raft_model.cuda()
    raft_model.eval()
    
    raft_iters = RAFT_config.iters
    
    # start benchmarking
    for corruption in corruption_name_list:
        if corruption == 'clean':
            print(f'Benchmarking [red]{corruption}[/red] ...')
            logging.info(f'{corruption} ...')
            
            # save_flow_path = f'../../outputs/{project_name}/flow_images/{corruption}'
            # if os.path.exists(save_flow_path):
            #     shutil.rmtree(save_flow_path)
            # os.makedirs(save_flow_path, exist_ok=True)
            
            results = benchmark_GoPro_C(data_root, split, corruption, 0, raft_model, raft_iters, save_flow_path)
            
            print(f'[red]{corruption}[/red]: {results}')
            logging.info(f'{corruption}: {results}')
            
            # amp = results['amp']
            # amp = np.concatenate(amp)
            # print(amp.shape)
            # np.save(f'./amp_gopro_10.npy', amp)
            
            for ik in result_terms:
                all_result_dict[corruption][ik].append(results[ik])
        
        else:
            for severity in severity_list:
                print(f'Benchmarking [red]{corruption}[/red] at level {severity} ...')
                logging.info(f'{corruption} at level {severity} ...')
                
                # save_flow_path = f'../../outputs/{project_name}/flow_images/{corruption}_{severity}'
                # if os.path.exists(save_flow_path):
                #     shutil.rmtree(save_flow_path)
                # os.makedirs(save_flow_path, exist_ok=True)
                
                results = benchmark_GoPro_C(data_root, split, corruption, severity, raft_model, raft_iters, save_flow_path)
                
                print(f'[red]{corruption}@{severity}[/red]: {results}')
                logging.info(f'{corruption}@{severity}: {results}')
                
                for ik in result_terms:
                    all_result_dict[corruption][ik].append(results[ik])
                    
        
    for k in all_result_dict.keys():
        for ik in result_terms:
            all_result_dict[k][f'{ik}_mean'] = np.array(all_result_dict[k][ik]).mean()
        all_result_dict[k]['result'] = f'{all_result_dict[k]["rCRE_mean"]:.4f}'

    print(all_result_dict)
    
    result_json = json.dumps(all_result_dict, indent=4)
    with open(f'./benchmark_results/{project_name}/results.json', 'w') as f:
        f.write(result_json)
    
    logging.info('------------')
    logging.info(result_json)
