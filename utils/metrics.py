"""
Evaluation Metrics
===================
Standard trajectory prediction metrics:
- ADE: Average Displacement Error
- FDE: Final Displacement Error
- minADE / minFDE: Best-of-K variants (standard for multi-modal)
"""

import torch
import numpy as np


def ade(predictions, ground_truth):
    """
    Average Displacement Error.
    
    Mean Euclidean distance between predicted and ground truth
    positions across all timesteps.
    
    Args:
        predictions:  (batch, pred_len, 2) or (pred_len, 2)
        ground_truth: (batch, pred_len, 2) or (pred_len, 2)
    
    Returns:
        ADE value (scalar)
    """
    displacement = torch.norm(predictions - ground_truth, dim=-1)  # (batch, pred_len) or (pred_len,)
    return displacement.mean().item()


def fde(predictions, ground_truth):
    """
    Final Displacement Error.
    
    Euclidean distance between the final predicted point and the
    actual final position.
    
    Args:
        predictions:  (batch, pred_len, 2) or (pred_len, 2)
        ground_truth: (batch, pred_len, 2) or (pred_len, 2)
    
    Returns:
        FDE value (scalar)
    """
    if predictions.dim() == 2:
        return torch.norm(predictions[-1] - ground_truth[-1]).item()
    return torch.norm(predictions[:, -1] - ground_truth[:, -1], dim=-1).mean().item()


def min_ade(predictions, ground_truth):
    """
    Minimum ADE across K modes (standard multi-modal metric).
    
    For each sample, reports the ADE of the closest prediction.
    
    Args:
        predictions:  (batch, K, pred_len, 2)
        ground_truth: (batch, pred_len, 2)
    
    Returns:
        minADE value (scalar)
        best_mode_indices: (batch,) index of best mode per sample
    """
    gt_expanded = ground_truth.unsqueeze(1)  # (batch, 1, pred_len, 2)
    displacement = torch.norm(predictions - gt_expanded, dim=-1)  # (batch, K, pred_len)
    ade_per_mode = displacement.mean(dim=-1)  # (batch, K)
    
    min_ade_vals, best_modes = ade_per_mode.min(dim=1)  # (batch,)
    
    return min_ade_vals.mean().item(), best_modes


def min_fde(predictions, ground_truth):
    """
    Minimum FDE across K modes.
    
    Args:
        predictions:  (batch, K, pred_len, 2)
        ground_truth: (batch, pred_len, 2)
    
    Returns:
        minFDE value (scalar)
        best_mode_indices: (batch,)
    """
    gt_expanded = ground_truth.unsqueeze(1)  # (batch, 1, pred_len, 2)
    final_disp = torch.norm(
        predictions[:, :, -1, :] - gt_expanded[:, :, -1, :], dim=-1
    )  # (batch, K)
    
    min_fde_vals, best_modes = final_disp.min(dim=1)  # (batch,)
    
    return min_fde_vals.mean().item(), best_modes


def most_likely_ade(predictions, mode_probs, ground_truth):
    """
    ADE of the most likely mode (highest probability).
    
    Args:
        predictions:  (batch, K, pred_len, 2)
        mode_probs:   (batch, K)
        ground_truth: (batch, pred_len, 2)
    
    Returns:
        ML-ADE value (scalar)
    """
    best_mode_idx = mode_probs.argmax(dim=1)  # (batch,)
    batch_idx = torch.arange(predictions.shape[0], device=predictions.device)
    best_pred = predictions[batch_idx, best_mode_idx]  # (batch, pred_len, 2)
    
    return ade(best_pred, ground_truth)


def most_likely_fde(predictions, mode_probs, ground_truth):
    """
    FDE of the most likely mode.
    
    Args:
        predictions:  (batch, K, pred_len, 2)
        mode_probs:   (batch, K)
        ground_truth: (batch, pred_len, 2)
    
    Returns:
        ML-FDE value (scalar)
    """
    best_mode_idx = mode_probs.argmax(dim=1)  # (batch,)
    batch_idx = torch.arange(predictions.shape[0], device=predictions.device)
    best_pred = predictions[batch_idx, best_mode_idx]  # (batch, pred_len, 2)
    
    return fde(best_pred, ground_truth)


def compute_all_metrics(predictions, mode_probs, ground_truth):
    """
    Compute all evaluation metrics.
    
    Args:
        predictions:  (batch, K, pred_len, 2)
        mode_probs:   (batch, K)
        ground_truth: (batch, pred_len, 2)
    
    Returns:
        dict of metric_name → value
    """
    min_ade_val, _ = min_ade(predictions, ground_truth)
    min_fde_val, _ = min_fde(predictions, ground_truth)
    ml_ade = most_likely_ade(predictions, mode_probs, ground_truth)
    ml_fde = most_likely_fde(predictions, mode_probs, ground_truth)
    
    # Per-mode ADE/FDE
    metrics = {
        "minADE": min_ade_val,
        "minFDE": min_fde_val,
        "ML-ADE": ml_ade,
        "ML-FDE": ml_fde,
    }
    
    # Individual mode ADEs
    gt_expanded = ground_truth.unsqueeze(1)
    displacement = torch.norm(predictions - gt_expanded, dim=-1)
    ade_per_mode = displacement.mean(dim=-1).mean(dim=0)  # (K,)
    for k in range(predictions.shape[1]):
        metrics[f"Mode{k}_ADE"] = ade_per_mode[k].item()
    
    # Mode probability distribution
    avg_probs = mode_probs.mean(dim=0)  # (K,)
    for k in range(predictions.shape[1]):
        metrics[f"Mode{k}_Prob"] = avg_probs[k].item()
    
    return metrics
