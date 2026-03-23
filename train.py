import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from config import Config
from models.social_transformer import SocialTransformer
from data.nuscenes_dataset import create_dataloaders
from utils.losses import TrajectoryPredictionLoss
from utils.metrics import compute_all_metrics
from utils.visualization import (
    plot_trajectory_prediction,
    plot_training_curves,
)

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_optimizer_and_scheduler(model, config):
    # 2. Very Low Learning Rate (strict 1e-6 as requested)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=config.WEIGHT_DECAY)
    
    # 8. Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    return optimizer, scheduler

def train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, config):
    """Train for one epoch."""
    model.train()
    epoch_losses = {
        "total": 0.0, "best_of_n": 0.0, "variety": 0.0, "mode_prob": 0.0
    }
    num_batches = 0
    total_grad_norm = 0.0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        obs_traj = batch["obs_traj"].to(device)
        obs_vel = batch["obs_vel"].to(device)
        obs_acc = batch["obs_acc"].to(device)
        obs_heading = batch["obs_heading"].to(device)
        obs_context = batch["obs_context"].to(device)
        pred_traj = batch["pred_traj"].to(device)
        neighbors_obs = batch["neighbors_obs"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)
        
        # 6. Normalize Input Data (zero mean, unit variance per batch)
        # Using batch statistics to normalize trajectory coordinates
        b_mean = obs_traj.mean(dim=(1, 2), keepdim=True)
        b_std = obs_traj.std(dim=(1, 2), keepdim=True) + 1e-6
        
        obs_traj_norm = (obs_traj - b_mean) / b_std
        pred_traj_norm = (pred_traj - b_mean) / b_std
        
        # Additionally normalizing velocities to keep inputs stable
        vel_mean = obs_vel.mean(dim=(1, 2), keepdim=True)
        vel_std = obs_vel.std(dim=(1, 2), keepdim=True) + 1e-6
        obs_vel_norm = (obs_vel - vel_mean) / vel_std

        optimizer.zero_grad()
        
        # 6. Debugging & Safety Checks
        if torch.isnan(obs_traj).any():
            print("\n[WARNING] NaN detected in inputs! Skipping batch.")
            continue
            
        with autocast(enabled=(device.type == "cuda")):
            predictions_norm, mode_probs, goals, social_attn = model(
                obs_traj_norm, obs_vel_norm, obs_acc, obs_heading, obs_context, 
                neighbors_obs, neighbor_mask
            )
            
            # Detect NaN in outputs
            if torch.isnan(predictions_norm).any():
                print("\n[WARNING] NaN detected in model outputs! Skipping batch.")
                continue
            
            # 4. Hard Clamp Outputs
            # Prevent predictions from exploring infinity, ensuring stable loss
            predictions_norm = torch.clamp(predictions_norm, min=-100.0, max=100.0)
            
            total_loss, loss_dict = loss_fn(predictions_norm, mode_probs, pred_traj_norm)
            
        # 5. Loss Safety Check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n[WARNING] Loss is NaN/Inf! Skipping batch.")
            optimizer.zero_grad()
            continue
            
        if total_loss.item() > 1e6 or total_loss.item() < -1e6:
            print(f"\n[WARNING] Loss out of safe bounds ({total_loss.item():.2e})! Skipping batch.")
            optimizer.zero_grad()
            continue

        # Backward pass
        if device.type == "cuda":
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
        else:
            total_loss.backward()

        # Track valid parameters for gradient metrics
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) > 0:
            total_norm_before = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0)
            
            if torch.isnan(total_norm_before) or torch.isinf(total_norm_before):
                print(f"\n[WARNING] Gradient is NaN/Inf! Skipping.")
                optimizer.zero_grad()
                continue
                
            # 1. Gradient Stability Fixes: strictly clamp to 5.0 instead of skipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            display_grad = total_norm_before.item()
        else:
            display_grad = 0.0
            
        if device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        total_grad_norm += display_grad
        
        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += loss_dict.get(key, 0.0)
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "grad": f"{display_grad:.2f}",
        })
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(num_batches, 1)
        
    epoch_losses["grad_norm"] = total_grad_norm / max(num_batches, 1)
    
    return epoch_losses


@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    """Validate the model and compute all metrics."""
    model.eval()
    
    val_losses = {"total": 0.0, "best_of_n": 0.0, "variety": 0.0, "mode_prob": 0.0}
    all_metrics = {}
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validating", leave=False):
        obs_traj = batch["obs_traj"].to(device)
        obs_vel = batch["obs_vel"].to(device)
        obs_acc = batch["obs_acc"].to(device)
        obs_heading = batch["obs_heading"].to(device)
        obs_context = batch["obs_context"].to(device)
        pred_traj = batch["pred_traj"].to(device)
        neighbors_obs = batch["neighbors_obs"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)
        
        # Normalize validation inputs identically to training
        b_mean = obs_traj.mean(dim=(1, 2), keepdim=True)
        b_std = obs_traj.std(dim=(1, 2), keepdim=True) + 1e-6
        obs_traj_norm = (obs_traj - b_mean) / b_std
        pred_traj_norm = (pred_traj - b_mean) / b_std
        
        vel_mean = obs_vel.mean(dim=(1, 2), keepdim=True)
        vel_std = obs_vel.std(dim=(1, 2), keepdim=True) + 1e-6
        obs_vel_norm = (obs_vel - vel_mean) / vel_std
        
        predictions_norm, mode_probs, goals, social_attn = model(
            obs_traj_norm, obs_vel_norm, obs_acc, obs_heading, obs_context, 
            neighbors_obs, neighbor_mask
        )
        
        predictions_norm = torch.clamp(predictions_norm, min=-100.0, max=100.0)
        total_loss, loss_dict = loss_fn(predictions_norm, mode_probs, pred_traj_norm)
        
        # Un-normalize predictions so ADE/FDE metrics are in true physical meters!
        predictions_meters = (predictions_norm * b_std.unsqueeze(1)) + b_mean.unsqueeze(1)
        
        for key in val_losses:
            val_losses[key] += loss_dict.get(key, 0.0)
        
        # Compute metrics in true physical scale
        batch_metrics = compute_all_metrics(predictions_meters, mode_probs, pred_traj)
        for key, val in batch_metrics.items():
            all_metrics[key] = all_metrics.get(key, 0.0) + val
        
        num_batches += 1
    
    for key in val_losses:
        val_losses[key] /= max(num_batches, 1)
    for key in all_metrics:
        all_metrics[key] /= max(num_batches, 1)
    
    return val_losses, all_metrics


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, filename="best_model.pt"):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)
    print(f"  [Checkpoint] Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train SocialTransformer")
    parser.add_argument("--nuscenes", action="store_true", help="Use real nuScenes data")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    args = parser.parse_args()
    
    config = Config()
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    config.ensure_dirs()
    
    set_seed(config.SEED)
    
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[CPU] Using CPU")
    
    print("\n[Data] Loading data...")
    use_synthetic = not args.nuscenes
    train_loader, val_loader, test_loader = create_dataloaders(
        config, use_synthetic=use_synthetic
    )
    
    print("\n[Model] Building SocialTransformer model from scratch...")
    model = SocialTransformer(config).to(device)
    
    # Initialize with clean weights to ensure stability
    model.apply(init_weights)
    
    loss_fn = TrajectoryPredictionLoss(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    best_min_ade = float("inf")
    start_epoch = 0
    resume_path = os.path.join(config.CHECKPOINT_DIR, "best_model_stable.pt")
    if os.path.exists(resume_path):
        print(f"\n[Checkpoint] Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # In stable training, sometimes we want to reset optimizer or keep it. Let's keep it.
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        if "metrics" in checkpoint and "minADE" in checkpoint["metrics"]:
            best_min_ade = checkpoint["metrics"]["minADE"]
            
    print(f"\n{'='*60}")
    print(f"  Starting/Resuming STABLE training: {config.NUM_EPOCHS} epochs (from epoch {start_epoch})")
    print(f"  Optimizer: Adam | LR: 1e-6 | Grad Clip: 5.0")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"{'='*60}\n")
    train_loss_history = []
    val_loss_history = []
    val_metric_history = []
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()
        
        train_losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, config
        )
        
        val_losses, val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Step the plateau scheduler with validation ADE
        val_ade = val_metrics.get("minADE", float("inf"))
        scheduler.step(val_ade)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Logging per epoch
        print(
            f"Epoch [{epoch+1:3d}/{config.NUM_EPOCHS}] "
            f"| Train Loss: {train_losses['total']:.4f} "
            f"| Val Loss: {val_losses['total']:.4f} "
            f"| minADE: {val_ade:.4f} "
            f"| minFDE: {val_metrics.get('minFDE', 0):.4f} "
            f"| LR: {current_lr:.6e} "
            f"| Avg Grad: {train_losses.get('grad_norm', 0):.2f} "
            f"| Time: {epoch_time:.1f}s"
        )
        
        train_loss_history.append(train_losses["total"])
        val_loss_history.append(val_losses["total"])
        val_metric_history.append(val_metrics)
        
        if val_ade < best_min_ade:
            best_min_ade = val_ade
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics,
                config.CHECKPOINT_DIR, "best_model_stable.pt"
            )

    print("\n[Done] Training complete!")
    print(f"  Best minADE: {best_min_ade:.4f} m")

if __name__ == "__main__":
    main()
