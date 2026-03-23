"""
Loss Functions for Multi-Modal Trajectory Prediction
======================================================
- Best-of-N Loss: Only backprops through the closest prediction mode
- Variety Loss: Penalizes mode collapse (encourages diverse predictions)
- Mode Loss: Cross-entropy on mode probability prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ade(predictions, ground_truth):
    """
    Average Displacement Error per mode.
    
    Args:
        predictions:  (batch, K, pred_len, 2)
        ground_truth: (batch, pred_len, 2)
    
    Returns:
        ade: (batch, K) ADE for each mode
    """
    gt_expanded = ground_truth.unsqueeze(1)  # (batch, 1, pred_len, 2)
    displacement = torch.norm(predictions - gt_expanded, dim=-1)  # (batch, K, pred_len)
    ade = displacement.mean(dim=-1)  # (batch, K)
    return ade


def compute_fde(predictions, ground_truth):
    """
    Final Displacement Error per mode.
    
    Args:
        predictions:  (batch, K, pred_len, 2)
        ground_truth: (batch, pred_len, 2)
    
    Returns:
        fde: (batch, K) FDE for each mode
    """
    gt_expanded = ground_truth.unsqueeze(1)  # (batch, 1, pred_len, 2)
    final_disp = torch.norm(
        predictions[:, :, -1, :] - gt_expanded[:, :, -1, :], dim=-1
    )  # (batch, K)
    return final_disp


class BestOfNLoss(nn.Module):
    """
    Best-of-N (Winner-Takes-All) Loss.
    
    For multi-modal predictions, only backpropagates through the mode
    that is closest to the ground truth. This encourages each mode to
    specialize in different trajectory patterns.
    
    Loss = min_k(ADE(pred_k, gt))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, ground_truth):
        """
        Args:
            predictions:  (batch, K, pred_len, 2) predicted trajectories
            ground_truth: (batch, pred_len, 2) actual future
        
        Returns:
            loss: scalar
            best_mode_idx: (batch,) indices of best mode per sample
        """
        ade = compute_ade(predictions, ground_truth)  # (batch, K)
        
        # Find best mode (lowest ADE)
        min_ade, best_mode_idx = ade.min(dim=1)  # (batch,)
        
        # Loss is the ADE of the best mode
        loss = min_ade.mean()
        
        return loss, best_mode_idx


class VarietyLoss(nn.Module):
    """
    Variety Loss to prevent mode collapse.
    
    Encourages the K predictions to be diverse by penalizing when
    modes are too close to each other.
    
    Loss = -mean(min_distance_between_modes)
    """
    
    def __init__(self, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
    
    def forward(self, predictions):
        """
        Args:
            predictions: (batch, K, pred_len, 2)
        
        Returns:
            loss: scalar (negative, to be minimized → maximize diversity)
        """
        batch, K, pred_len, _ = predictions.shape
        
        if K <= 1:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute pairwise distances between mode endpoints
        endpoints = predictions[:, :, -1, :]  # (batch, K, 2)
        
        total_dist = 0
        n_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist = torch.norm(endpoints[:, i] - endpoints[:, j], dim=-1)
                # Bounded Variety Loss: minimize e^(-dist) instead of unbounded -dist
                # This prevents the predictions from exploring infinity to get a huge negative loss
                # It forces distance to increase but the reward diminishes and is bounded between [0, 1]
                total_dist = total_dist + torch.exp(-dist).mean()
                n_pairs += 1
                
        variety_loss = total_dist / max(n_pairs, 1)
        
        return variety_loss


class ModeProbabilityLoss(nn.Module):
    """
    Cross-entropy loss on mode probability predictions.
    
    The target is the mode closest to ground truth (from BestOfN).
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, mode_probs, best_mode_idx):
        """
        Args:
            mode_probs:    (batch, K) predicted probabilities
            best_mode_idx: (batch,) index of the actual best mode
        
        Returns:
            loss: scalar
        """
        # Small epsilon to prevent log(0) which causes -Inf and exploding gradients
        eps = 1e-6
        # Clamp probabilities carefully to stay in valid range before log
        clamped_probs = torch.clamp(mode_probs, min=eps, max=1.0-eps)
        
        # Cross entropy of probabilities is -log(P_target)
        # Select the probability of the mode that is closest to ground truth
        batch_idx = torch.arange(mode_probs.shape[0], device=mode_probs.device)
        target_probs = clamped_probs[batch_idx, best_mode_idx]
        
        # Compute negative log likelihood
        nll_loss = -torch.log(target_probs)
        
        return nll_loss.mean()


class TrajectoryPredictionLoss(nn.Module):
    """
    Combined loss for training.
    
    Total Loss = BestOfN + λ_variety * Variety + λ_mode * Mode
    """
    
    def __init__(self, config):
        super().__init__()
        self.best_of_n = BestOfNLoss()
        self.variety = VarietyLoss(config.NUM_MODES)
        self.mode_prob = ModeProbabilityLoss()
        
        self.variety_weight = config.VARIETY_LOSS_WEIGHT
        self.mode_weight = config.MODE_LOSS_WEIGHT
    
    def forward(self, predictions, mode_probs, ground_truth):
        """
        Compute combined loss.
        
        Args:
            predictions:  (batch, K, pred_len, 2)
            mode_probs:   (batch, K)
            ground_truth: (batch, pred_len, 2)
        
        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses for logging
        """
        # Best-of-N trajectory loss
        bon_loss, best_mode_idx = self.best_of_n(predictions, ground_truth)
        
        # Variety loss
        var_loss = self.variety(predictions)
        
        # Mode probability loss
        mode_loss = self.mode_prob(mode_probs, best_mode_idx)
        
        # Combined
        total_loss = (
            bon_loss
            + self.variety_weight * var_loss
            + self.mode_weight * mode_loss
        )
        
        
        # Catch NaNs/Infs immediately in the loss step
        if not torch.isfinite(total_loss):
            print(f"\n[WARNING] Loss is not finite! bon_loss={bon_loss.item():.4f}, var_loss={var_loss.item():.4f}, mode_loss={mode_loss.item():.4f}")
            # Replace with a safe dummy loss gradient, or 0 to let the safety check in train step skip
            total_loss = torch.tensor(0.0, requires_grad=True, device=predictions.device)
            
        loss_dict = {
            "total": total_loss.item(),
            "best_of_n": bon_loss.item(),
            "variety": var_loss.item(),
            "mode_prob": mode_loss.item(),
        }
        
        return total_loss, loss_dict
