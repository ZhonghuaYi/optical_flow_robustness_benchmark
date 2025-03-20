import torch 
import numpy as np


def EPE(pred_flow: torch.Tensor, gt_flow: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """
    Compute the End Point Error (EPE) between the predicted flow and the ground truth flow.
    
    Args:
    - pred_flow (torch.Tensor): Predicted flow tensor of shape [B, 2, H, W]
    - gt_flow (torch.Tensor): Ground truth flow tensor of shape [B, 2, H, W]
    - mask (torch.Tensor): Mask tensor of shape [B, 1, H, W] indicating the valid regions. Default: None
    
    Returns:
    - epe (torch.Tensor): End Point Error tensor of shape [B*H*W]
    """
    assert pred_flow.shape == gt_flow.shape, 'Input tensors must have the same shape'
    if mask is not None:
        assert mask.shape[0] == pred_flow.shape[0] and mask.shape[-2:] == pred_flow.shape[-2:], f'Mask shape {mask.shape} must have the same B, H, W as input tensors {pred_flow.shape}'
    
    epe = torch.norm(pred_flow - gt_flow, p=2, dim=1) # [B, H, W]
    epe = epe.view(-1) # [B*H*W]
    if mask is not None:
        mask = mask.view(-1) > 0
        epe = epe[mask]
    
    return epe.mean().view(1)


def CRE(pred_clean_flow: torch.Tensor, pred_corrupt_flow: torch.Tensor, gt_flow: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """
    Compute the Corruption Robustness Error (CRE) between the predicted clean flow and the predicted corrupted flow, with respect to the ground truth flow.
    
    Args:
    - pred_clean_flow (torch.Tensor): Predicted clean flow tensor of shape [B, 2, H, W]
    - pred_corrupt_flow (torch.Tensor): Predicted corrupted flow tensor of shape [B, 2, H, W]
    - gt_flow (torch.Tensor): Ground truth flow tensor of shape [B, 2, H, W]
    - mask (torch.Tensor): Mask tensor of shape [B, 1, H, W] indicating the valid regions. Default: None
    
    Returns:
    - cre (torch.Tensor): Corruption Robustness Error tensor of shape [B*H*W]
    """
    assert pred_clean_flow.shape == pred_corrupt_flow.shape == gt_flow.shape, 'Input tensors must have the same shape'
    if mask is not None:
        assert mask.shape[0] == pred_clean_flow.shape[0] and mask.shape[-2:] == pred_clean_flow.shape[-2:], f'Mask shape {mask.shape} must have the same B, H, W as input tensors {pred_clean_flow.shape}'
    
    clean_epe = torch.norm(pred_clean_flow - gt_flow, p=2, dim=1) # [B, H, W]
    corrupt_epe = torch.norm(pred_corrupt_flow - gt_flow, p=2, dim=1) # [B, H, W]
    
    cre = corrupt_epe - clean_epe # [B, H, W]
    cre = cre.view(-1) # [B*H*W]
    if mask is not None:
        mask = mask.view(-1) > 0
        cre = cre[mask]
    
    return cre.mean().view(1)


def rCRE(pred_clean_flow: torch.Tensor, pred_corrupt_flow: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """
    Compute the Relative Corruption Robustness Error (rCRE) between the predicted clean flow and the predicted corrupted flow, without ground truth flow.
    
    Args:
    - pred_clean_flow (torch.Tensor): Predicted clean flow tensor of shape [B, 2, H, W]
    - pred_corrupt_flow (torch.Tensor): Predicted corrupted flow tensor of shape [B, 2, H, W]
    - mask (torch.Tensor): Mask tensor of shape [B, 1, H, W] indicating the valid regions. Default: None
    
    Returns:
    - rcre (torch.Tensor): Relative Corruption Robustness Error tensor of shape [B*H*W]
    """
    assert pred_clean_flow.shape == pred_corrupt_flow.shape, 'Input tensors must have the same shape'
    if mask is not None:
        assert mask.shape[0] == pred_clean_flow.shape[0] and mask.shape[-2:] == pred_clean_flow.shape[-2:], f'Mask shape {mask.shape} must have the same B, H, W as input tensors {pred_clean_flow.shape}'
    
    rcre = torch.norm(pred_corrupt_flow - pred_clean_flow, p=2, dim=1) # [B, H, W]
    rcre = rcre.view(-1) # [B*H*W]
    if mask is not None:
        mask = mask.view(-1) > 0
        rcre = rcre[mask]
    
    return rcre.mean().view(1)
    
