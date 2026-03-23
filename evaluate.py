"""
Evaluation Script
==================
Comprehensive evaluation of a trained SocialTransformer model.

Computes:
- ADE, FDE (single best mode)
- minADE, minFDE (best of K modes)
- ML-ADE, ML-FDE (most likely mode)
- Per-mode statistics
- Qualitative trajectory visualizations
- Social attention analysis

Usage:
    python evaluate.py                           # Eval with synthetic data
    python evaluate.py --nuscenes                # Eval with real nuScenes
    python evaluate.py --checkpoint path/to/ckpt # Custom checkpoint
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

from config import Config
from models.social_transformer import SocialTransformer
from data.nuscenes_dataset import create_dataloaders
from utils.losses import TrajectoryPredictionLoss
from utils.metrics import compute_all_metrics, min_ade, min_fde
from utils.visualization import (
    plot_trajectory_prediction,
    plot_social_attention,
)


def evaluate_model(model, test_loader, device, config, save_dir):
    """
    Full model evaluation with all metrics and visualizations.
    """
    model.eval()
    
    all_metrics_accum = {}
    all_ades = []    # per-sample minADE
    all_fdes = []    # per-sample minFDE
    num_batches = 0
    
    print("[Data] Computing metrics on test set...")
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        obs_traj = batch["obs_traj"].to(device)
        obs_vel = batch["obs_vel"].to(device)
        obs_acc = batch["obs_acc"].to(device)
        obs_heading = batch["obs_heading"].to(device)
        obs_context = batch["obs_context"].to(device)
        pred_traj = batch["pred_traj"].to(device)
        neighbors_obs = batch["neighbors_obs"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)
        
        with torch.no_grad():
            predictions, mode_probs, goals, social_attn = model(
                obs_traj, obs_vel, obs_acc, obs_heading, obs_context, 
                neighbors_obs, neighbor_mask
            )
        
        # Compute batch metrics
        batch_metrics = compute_all_metrics(predictions, mode_probs, pred_traj)
        for key, val in batch_metrics.items():
            all_metrics_accum[key] = all_metrics_accum.get(key, 0.0) + val
        
        # Per-sample minADE for distribution analysis
        gt_expanded = pred_traj.unsqueeze(1)
        displacement = torch.norm(predictions - gt_expanded, dim=-1)
        ade_per_mode = displacement.mean(dim=-1)
        min_ade_per_sample = ade_per_mode.min(dim=1)[0]
        all_ades.extend(min_ade_per_sample.cpu().numpy().tolist())
        
        # Per-sample minFDE
        final_disp = torch.norm(
            predictions[:, :, -1, :] - gt_expanded[:, :, -1, :], dim=-1
        )
        min_fde_per_sample = final_disp.min(dim=1)[0]
        all_fdes.extend(min_fde_per_sample.cpu().numpy().tolist())
        
        num_batches += 1
    
    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in all_metrics_accum.items()}
    
    return avg_metrics, all_ades, all_fdes


def generate_visualizations(model, test_loader, device, save_dir, num_samples=15):
    """Generate detailed trajectory visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    print(f"\n[Viz] Generating {num_samples} trajectory visualizations...")
    
    sample_count = 0
    for batch in test_loader:
        obs_traj = batch["obs_traj"].to(device)
        obs_vel = batch["obs_vel"].to(device)
        obs_acc = batch["obs_acc"].to(device)
        obs_heading = batch["obs_heading"].to(device)
        obs_context = batch["obs_context"].to(device)
        pred_traj = batch["pred_traj"].to(device)
        neighbors_obs = batch["neighbors_obs"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)
        
        with torch.no_grad():
            predictions, mode_probs, goals, social_attn = model(
                obs_traj, obs_vel, obs_acc, obs_heading, obs_context, 
                neighbors_obs, neighbor_mask
            )
        
        for i in range(min(obs_traj.shape[0], num_samples - sample_count)):
            # Trajectory prediction plot
            plot_trajectory_prediction(
                obs_traj=obs_traj[i],
                pred_traj_gt=pred_traj[i],
                pred_traj_modes=predictions[i],
                mode_probs=mode_probs[i],
                neighbors_obs=neighbors_obs[i],
                neighbor_mask=neighbor_mask[i],
                title=f"Test Sample {sample_count + 1}",
                save_path=os.path.join(
                    save_dir, f"trajectory_{sample_count + 1}.png"
                ),
            )
            
            # Social attention plot (if neighbors exist)
            if neighbor_mask[i].any():
                plot_social_attention(
                    obs_traj=obs_traj[i],
                    neighbors_obs=neighbors_obs[i],
                    neighbor_mask=neighbor_mask[i],
                    attention_weights=social_attn[i],
                    save_path=os.path.join(
                        save_dir, f"social_attn_{sample_count + 1}.png"
                    ),
                    title=f"Social Attention — Sample {sample_count + 1}",
                )
            
            sample_count += 1
            if sample_count >= num_samples:
                break
        
        if sample_count >= num_samples:
            break
    
    print(f"  ✅ {sample_count} visualizations saved to {save_dir}")


def generate_error_distribution(all_ades, all_fdes, save_dir):
    """Plot error distribution histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#FFFFFF")
    
    # ADE distribution
    axes[0].hist(
        all_ades, bins=50, color="#3498DB", alpha=0.7, edgecolor="white"
    )
    axes[0].axvline(
        np.mean(all_ades), color="#E74C3C", linestyle="--",
        linewidth=2, label=f"Mean: {np.mean(all_ades):.3f}m"
    )
    axes[0].axvline(
        np.median(all_ades), color="#2ECC71", linestyle="--",
        linewidth=2, label=f"Median: {np.median(all_ades):.3f}m"
    )
    axes[0].set_xlabel("minADE (meters)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("minADE Distribution", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # FDE distribution
    axes[1].hist(
        all_fdes, bins=50, color="#E74C3C", alpha=0.7, edgecolor="white"
    )
    axes[1].axvline(
        np.mean(all_fdes), color="#3498DB", linestyle="--",
        linewidth=2, label=f"Mean: {np.mean(all_fdes):.3f}m"
    )
    axes[1].axvline(
        np.median(all_fdes), color="#2ECC71", linestyle="--",
        linewidth=2, label=f"Median: {np.median(all_fdes):.3f}m"
    )
    axes[1].set_xlabel("minFDE (meters)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("minFDE Distribution", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_dir, "error_distribution.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    
    # Percentile analysis
    percentiles = [50, 75, 90, 95, 99]
    print("\n[Stats] Error Percentiles:")
    print(f"  {'Percentile':>12s} | {'minADE (m)':>10s} | {'minFDE (m)':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")
    for p in percentiles:
        ade_p = np.percentile(all_ades, p)
        fde_p = np.percentile(all_fdes, p)
        print(f"  {p:>11d}% | {ade_p:>10.4f} | {fde_p:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SocialTransformer")
    parser.add_argument("--nuscenes", action="store_true", help="Use real nuScenes")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: best_model.pt)"
    )
    parser.add_argument(
        "--num_viz", type=int, default=15,
        help="Number of visualization samples"
    )
    args = parser.parse_args()
    
    config = Config()
    config.ensure_dirs()
    
    # Device
    device = torch.device(
        "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[Device] Device: {device}")
    
    # Data
    use_synthetic = not args.nuscenes
    _, _, test_loader = create_dataloaders(config, use_synthetic=use_synthetic)
    
    # Model
    model = SocialTransformer(config).to(device)
    
    # Load checkpoint
    ckpt_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(ckpt_path):
        print(f"[Loading] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
    else:
        print(f"⚠️  No checkpoint found at {ckpt_path}")
        print("  Running evaluation with untrained model (random weights)...")
    
    # Evaluate
    eval_dir = os.path.join(config.RESULTS_DIR, "evaluation")
    avg_metrics, all_ades, all_fdes = evaluate_model(
        model, test_loader, device, config, eval_dir
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  minADE:  {avg_metrics['minADE']:8.4f} m           │")
    print(f"  │  minFDE:  {avg_metrics['minFDE']:8.4f} m           │")
    print(f"  │  ML-ADE:  {avg_metrics['ML-ADE']:8.4f} m           │")
    print(f"  │  ML-FDE:  {avg_metrics['ML-FDE']:8.4f} m           │")
    print(f"  └─────────────────────────────────┘")
    
    print(f"\n  Mode Statistics:")
    for k in range(config.NUM_MODES):
        ade_key = f"Mode{k}_ADE"
        prob_key = f"Mode{k}_Prob"
        if ade_key in avg_metrics:
            print(f"    Mode {k}: ADE={avg_metrics[ade_key]:.4f}m, "
                  f"Prob={avg_metrics[prob_key]:.3f}")
    
    # Error distribution
    generate_error_distribution(all_ades, all_fdes, eval_dir)
    
    # Visualizations
    viz_dir = os.path.join(eval_dir, "visualizations")
    generate_visualizations(model, test_loader, device, viz_dir, args.num_viz)
    
    # Save metrics
    metrics_path = os.path.join(eval_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\n  [Done] Metrics saved to {metrics_path}")
    
    print("\n[Done] Evaluation complete!")


if __name__ == "__main__":
    main()
