"""
Hackathon Demo Evaluation & Visualization Pipeline
==================================================
1. Evaluates best_model_stable.pt
2. Computes and prints precise minADE & minFDE
3. Hunts for "Interesting" samples (large movement / sharp turns)
4. Exports presentation-ready plots mapping multi-modal future trajectories.
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import Config
from models.social_transformer import SocialTransformer
from data.nuscenes_dataset import create_dataloaders
from utils.metrics import compute_all_metrics

# -----------------
# NEW: Behavior Classification
# -----------------
def classify_behavior(obs, gt):
    """
    Classify the trajectory behavior into simple heuristics.
    """
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
        
    total_traj = np.vstack([obs, gt])
    start, end = total_traj[0], total_traj[-1]
    
    # Compute straight-line displacement
    displacement = np.linalg.norm(end - start)
    
    # Compute path curvature (path length / straight-line distance)
    diffs = total_traj[1:] - total_traj[:-1]
    path_len = np.sum(np.linalg.norm(diffs, axis=-1))
    
    curviness = path_len / (displacement + 1e-6)
    
    if curviness > 1.2:
        return "Complex Motion"
    elif curviness > 1.05:
        return "Turning"
    else:
        return "Straight"

# -----------------
# 1. Visualization
# -----------------
def plot_trajectory(obs, gt, modes, probs, sample_id, out_dir, is_failure=False):
    """
    Generate a presentation-ready plot for a single given scenario.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300) # dpi=300 for premium Quality
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f9fa')

    # Convert tensors to numpy if they aren't already
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    if isinstance(modes, torch.Tensor):
        modes = modes.cpu().numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    behavior = classify_behavior(obs, gt)

    # Past Trajectory (Blue)
    ax.plot(obs[:, 0], obs[:, 1], color='#2980b9', linewidth=3.5, marker='o', 
            markersize=6, label='Observed History', zorder=3)
            
    # Start Point (Distinct Marker)
    ax.scatter(obs[0, 0], obs[0, 1], color='#1abc9c', s=150, marker='D', edgecolor='white', linewidth=1.5, zorder=5, label='Start Point')

    # Ground Truth Future (Green)
    gt_conn = np.vstack([obs[-1:], gt])
    ax.plot(gt_conn[:, 0], gt_conn[:, 1], color='#27ae60', linewidth=4.5, 
            linestyle='-', marker='s', markersize=6, label='Ground Truth Future', zorder=2)
            
    # End Point Marker
    ax.scatter(gt[-1, 0], gt[-1, 1], color='#27ae60', s=200, marker='*', edgecolor='white', linewidth=1.5, zorder=5, label='GT End Point')

    # NEW: Highlight Best Mode
    best_mode_idx = np.argmax(probs)

    # Predicted Multi-Modal Futures (Red/Orange hue)
    for k in range(modes.shape[0]):
        pred = modes[k]
        prob = probs[k]
        
        is_best = (k == best_mode_idx)
        
        # Scale alpha and line thickness by the model's confidence probability
        alpha = 0.9 if is_best else max(0.2, float(prob))
        lw = 3.5 if is_best else (1.0 + (prob * 2.0))
        color = '#8b0000' if is_best else '#e74c3c' # Dark red for best mode
        ls = '-' if is_best else '--'
        
        pred_conn = np.vstack([obs[-1:], pred])
        label = f'Pred Mode {k+1} (p={prob:.2f}){" [BEST]" if is_best else ""}'
        
        ax.plot(pred_conn[:, 0], pred_conn[:, 1], color=color, linewidth=lw, 
                alpha=alpha, linestyle=ls, marker='X', markersize=4, label=label, zorder=4 if is_best else 1)
        ax.scatter(pred[-1, 0], pred[-1, 1], color=color, s=120 if is_best else 80, alpha=alpha, zorder=4)

        # NEW: Ellipse Visualization
        # Estimate uncertainty based on probability (lower prob -> higher uncertainty -> larger ellipse)
        variance = max(0.5, (1.0 - prob) * 3.0) 
        width = variance * 1.5
        height = variance * 1.5
        
        ellipse = patches.Ellipse(
            (pred[-1, 0], pred[-1, 1]), 
            width=width, 
            height=height, 
            angle=0, 
            alpha=alpha * 0.3, # Highly transparent
            facecolor=color, 
            edgecolor=color, 
            linewidth=1.5,
            zorder=1
        )
        ax.add_patch(ellipse)

    # Clean layout & Title
    if is_failure:
        title = f"Failure Case \u2013 Model Uncertain\nScenario #{sample_id} - {behavior}"
    else:
        title = f"Trajectory Prediction Analysis\nScenario #{sample_id} - {behavior}"
        
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel("X Transform (Meters)", fontsize=12, fontweight='bold', color='#34495e')
    ax.set_ylabel("Y Transform (Meters)", fontsize=12, fontweight='bold', color='#34495e')
    
    # Legend - cleaner outside plot
    handles, labels = ax.get_legend_handles_labels()
    # Filter duplicate labels if any
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), 
              borderaxespad=0., framealpha=0.9, edgecolor='#2c3e50', fontsize=10)
    
    # Grid & Scaling
    ax.grid(True, linestyle='--', alpha=0.5, color='#bdc3c7')
    ax.set_aspect('equal', adjustable='datalim') # Ensure realistic physical scaling
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if is_failure:
        out_path = os.path.join(out_dir, "failure_case.png")
    else:
        out_path = os.path.join(out_dir, f"scenario_analysis_{sample_id}.png")
        
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# -----------------
# 2. Score Function
# -----------------
def get_interestingness_score(obs, gt):
    """
    Heuristic to find 'cool' scenarios for presentation.
    Large displacements or noticeable curves score higher.
    """
    total_traj = torch.cat([obs, gt], dim=0)
    # 1. Total displacement length
    start, end = total_traj[0], total_traj[-1]
    displacement = torch.norm(end - start).item()
    
    # 2. Turn factor (Deviation from a straight line)
    path_len = torch.norm(total_traj[1:] - total_traj[:-1], dim=-1).sum().item()
    curviness = path_len / (displacement + 1e-6) # >1 means curved path
    
    return displacement * curviness

# -----------------
# 3. Main Eval
# -----------------
def evaluate_model():
    print("="*60)
    print("\U0001f680 Initializing Hackathon Demo Evaluation")
    print("="*60)
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Settings & Paths
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model_stable.pt")
    viz_out_dir = os.path.join(config.RESULTS_DIR, "outputs", "visualizations")
    os.makedirs(viz_out_dir, exist_ok=True)
    
    # Load Model
    model = SocialTransformer(config).to(device)
    if not os.path.exists(ckpt_path):
         raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"\u2705 Loaded checkpoint from Epoch {checkpoint.get('epoch', 'Unknown')}\n")

    # Load test data
    _, _, test_loader = create_dataloaders(config, use_synthetic=False) # True if you don't have NuScenes set up
    print(f"\u2705 Loaded Test DataLoader ({len(test_loader)} batches)")

    # Metrics Tracking
    all_metrics = {"minADE": 0.0, "minFDE": 0.0}
    num_batches = 0
    
    # Harvesting for 'interesting' plots
    scenario_candidates = []
    
    # NEW: Failure Case Logic - track worst FDE
    worst_fde = -1.0
    worst_scenario_data = None

    with torch.no_grad():
        for batch in test_loader:
            obs_traj = batch["obs_traj"].to(device)
            pred_traj = batch["pred_traj"].to(device)
            
            # Retrieve un-normalization targets
            b_mean = obs_traj.mean(dim=(1, 2), keepdim=True)
            b_std = obs_traj.std(dim=(1, 2), keepdim=True) + 1e-6
            
            # Predict
            modes, probs, _, _ = model(
                obs_traj, batch["obs_vel"].to(device), batch["obs_acc"].to(device), 
                batch["obs_heading"].to(device), batch["obs_context"].to(device), 
                batch["neighbors_obs"].to(device), batch["neighbor_mask"].to(device)
            )
            
            # Un-normalize for realistic metrics (meters)
            modes_meters = (modes * b_std.unsqueeze(1)) + b_mean.unsqueeze(1)
            
            # Compute Batch Metrics
            batch_metric = compute_all_metrics(modes_meters, probs, pred_traj)
            all_metrics["minADE"] += batch_metric["minADE"]
            all_metrics["minFDE"] += batch_metric["minFDE"]
            num_batches += 1
            
            # Search batch for interesting & failure scenarios
            for i in range(obs_traj.shape[0]):
                obs_i = obs_traj[i].clone()
                gt_i = pred_traj[i].clone()
                modes_i = modes_meters[i].clone()
                probs_i = probs[i].clone()
                
                # Interestingness score
                score = get_interestingness_score(obs_i, gt_i)
                scenario_candidates.append({
                    'score': score,
                    'obs': obs_i,
                    'gt': gt_i,
                    'modes': modes_i, 
                    'probs': probs_i
                })
                
                # NEW: Failure Case Logic
                # minFDE for this specific sample
                fde_per_mode = torch.norm(gt_i[-1].unsqueeze(0) - modes_i[:, -1, :], dim=-1)
                min_fde_i = torch.min(fde_per_mode).item()
                if min_fde_i > worst_fde:
                    worst_fde = min_fde_i
                    worst_scenario_data = {
                        'obs': obs_i,
                        'gt': gt_i,
                        'modes': modes_i,
                        'probs': probs_i
                    }

    # Average metrics
    minADE = all_metrics["minADE"] / num_batches
    minFDE = all_metrics["minFDE"] / num_batches
    print("\n" + "="*60)
    print("\U0001f4ca CHALLENGE METRICS LOGGED")
    print("="*60)
    print(f"  \U0001f3c1 minADE: {minADE:.4f} meters")
    print(f"  \U0001f3c1 minFDE: {minFDE:.4f} meters")
    print("="*60 + "\n")
    
    # Process and plot top interesting scenarios
    # Sort candidates high -> low
    scenario_candidates.sort(key=lambda x: x['score'], reverse=True)
    top_scenarios = scenario_candidates[:8] # Best 8 interesting samples
    print(f"\U0001f4f8 Generating plots for {len(top_scenarios)} highly dynamic 'Interesting' scenarios...")
    
    for idx, sample in enumerate(top_scenarios):
        plot_trajectory(
            obs=sample['obs'], 
            gt=sample['gt'], 
            modes=sample['modes'], 
            probs=sample['probs'], 
            sample_id=idx + 1, 
            out_dir=viz_out_dir
        )
        
    # NEW: Plot Failure Case
    if worst_scenario_data is not None:
        print(f"\U0001f6a8 Saving worst failure case (FDE={worst_fde:.2f}m)...")
        plot_trajectory(
            obs=worst_scenario_data['obs'], 
            gt=worst_scenario_data['gt'], 
            modes=worst_scenario_data['modes'], 
            probs=worst_scenario_data['probs'], 
            sample_id='FAIL', 
            out_dir=viz_out_dir,
            is_failure=True
        )
        
    print(f"\U0001f389 Success! Generated highly aesthetic presentations inside: {viz_out_dir}")

if __name__ == "__main__":
    evaluate_model()