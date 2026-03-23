"""
Visualization Utilities
========================
Trajectory plotting for training monitoring, evaluation, and 
qualitative analysis.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


# Color palette for trajectory modes
MODE_COLORS = [
    "#FF6B6B",  # Mode 0: Red
    "#4ECDC4",  # Mode 1: Teal
    "#FFE66D",  # Mode 2: Yellow
    "#A8E6CF",  # Mode 3: Green (if K > 3)
    "#B8B5FF",  # Mode 4: Purple
]

OBS_COLOR = "#2C3E50"       # Dark blue-gray for observed
GT_COLOR = "#27AE60"        # Green for ground truth
NEIGHBOR_COLOR = "#BDC3C7"  # Light gray for neighbors


def plot_trajectory_prediction(
    obs_traj,
    pred_traj_gt,
    pred_traj_modes,
    mode_probs=None,
    neighbors_obs=None,
    neighbor_mask=None,
    title="Trajectory Prediction",
    save_path=None,
    show_probs=True,
    figsize=(10, 8),
):
    """
    Plot a single trajectory prediction with multiple modes.
    
    Args:
        obs_traj:       (obs_len, 2) observed positions
        pred_traj_gt:   (pred_len, 2) ground truth future
        pred_traj_modes: (K, pred_len, 2) predicted trajectories
        mode_probs:     (K,) probabilities for each mode
        neighbors_obs:  (N, obs_len, 2) neighbor observed trajectories
        neighbor_mask:  (N,) boolean mask
        title:          plot title
        save_path:      path to save the figure
        show_probs:     whether to annotate mode probabilities
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")
    
    # Convert to numpy if tensors
    if hasattr(obs_traj, "numpy"):
        obs_traj = obs_traj.cpu().numpy()
    if hasattr(pred_traj_gt, "numpy"):
        pred_traj_gt = pred_traj_gt.cpu().numpy()
    if hasattr(pred_traj_modes, "numpy"):
        pred_traj_modes = pred_traj_modes.cpu().numpy()
    if mode_probs is not None and hasattr(mode_probs, "numpy"):
        mode_probs = mode_probs.cpu().numpy()
    
    # ── Plot neighbors ──
    if neighbors_obs is not None and neighbor_mask is not None:
        if hasattr(neighbors_obs, "numpy"):
            neighbors_obs = neighbors_obs.cpu().numpy()
        if hasattr(neighbor_mask, "numpy"):
            neighbor_mask = neighbor_mask.cpu().numpy()
        
        for i in range(len(neighbor_mask)):
            if neighbor_mask[i]:
                n_traj = neighbors_obs[i]
                ax.plot(
                    n_traj[:, 0], n_traj[:, 1],
                    color=NEIGHBOR_COLOR, linewidth=1.5, alpha=0.5,
                    linestyle="--", zorder=1,
                )
                ax.scatter(
                    n_traj[-1, 0], n_traj[-1, 1],
                    color=NEIGHBOR_COLOR, s=30, alpha=0.5,
                    marker="s", zorder=2,
                )
    
    # ── Plot observed trajectory ──
    ax.plot(
        obs_traj[:, 0], obs_traj[:, 1],
        color=OBS_COLOR, linewidth=2.5, label="Observed",
        marker="o", markersize=5, zorder=5,
    )
    ax.scatter(
        obs_traj[0, 0], obs_traj[0, 1],
        color=OBS_COLOR, s=80, marker="^", zorder=6,
        edgecolors="white", linewidth=1.5,
        label="Start",
    )
    
    # ── Plot ground truth ──
    # Connect last obs point to first GT point
    connection = np.vstack([obs_traj[-1:], pred_traj_gt[:1]])
    ax.plot(
        connection[:, 0], connection[:, 1],
        color=GT_COLOR, linewidth=2, linestyle="-", alpha=0.5,
    )
    ax.plot(
        pred_traj_gt[:, 0], pred_traj_gt[:, 1],
        color=GT_COLOR, linewidth=2.5, label="Ground Truth",
        marker="D", markersize=4, zorder=4,
    )
    ax.scatter(
        pred_traj_gt[-1, 0], pred_traj_gt[-1, 1],
        color=GT_COLOR, s=100, marker="*", zorder=6,
        edgecolors="white", linewidth=1.5,
    )
    
    # ── Plot predicted modes ──
    K = pred_traj_modes.shape[0]
    legend_handles = []
    
    for k in range(K):
        color = MODE_COLORS[k % len(MODE_COLORS)]
        pred = pred_traj_modes[k]
        
        # Determine alpha based on probability
        alpha = 0.6
        linewidth = 2.0
        if mode_probs is not None:
            alpha = max(0.3, min(1.0, mode_probs[k] * 2))
            linewidth = 1.5 + mode_probs[k] * 2.5
        
        # Connect from last obs
        connection = np.vstack([obs_traj[-1:], pred[:1]])
        ax.plot(
            connection[:, 0], connection[:, 1],
            color=color, linewidth=linewidth, linestyle="-",
            alpha=alpha * 0.5,
        )
        
        # Plot predicted trajectory
        label = f"Mode {k}"
        if mode_probs is not None and show_probs:
            label += f" (p={mode_probs[k]:.2f})"
        
        ax.plot(
            pred[:, 0], pred[:, 1],
            color=color, linewidth=linewidth, alpha=alpha,
            marker="o", markersize=3, label=label, zorder=3,
        )
        
        # Plot endpoint
        ax.scatter(
            pred[-1, 0], pred[-1, 1],
            color=color, s=80, marker="X", alpha=alpha,
            edgecolors="white", linewidth=1, zorder=5,
        )
    
    # ── Formatting ──
    ax.set_xlabel("X (meters)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y (meters)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(
        loc="upper left", fontsize=9,
        framealpha=0.9, edgecolor="#2C3E50",
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.close(fig)
    return fig


def plot_training_curves(
    train_losses,
    val_losses,
    val_metrics,
    save_dir,
):
    """
    Plot training loss curves and validation metrics over epochs.
    
    Args:
        train_losses: list of train loss per epoch
        val_losses:   list of val loss per epoch
        val_metrics:  list of metric dicts per epoch
        save_dir:     directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    # ── Loss curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#FFFFFF")
    
    # Left: Loss
    axes[0].plot(epochs, train_losses, label="Train", color="#E74C3C", linewidth=2)
    axes[0].plot(epochs, val_losses, label="Val", color="#3498DB", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Right: ADE / FDE
    if val_metrics:
        min_ades = [m.get("minADE", 0) for m in val_metrics]
        min_fdes = [m.get("minFDE", 0) for m in val_metrics]
        
        axes[1].plot(epochs[:len(min_ades)], min_ades, label="minADE", color="#E74C3C", linewidth=2)
        axes[1].plot(epochs[:len(min_fdes)], min_fdes, label="minFDE", color="#2ECC71", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Error (meters)", fontsize=12)
        axes[1].set_title("Validation Metrics", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {save_dir}")


def plot_social_attention(
    obs_traj,
    neighbors_obs,
    neighbor_mask,
    attention_weights,
    save_path=None,
    title="Social Attention Visualization",
):
    """
    Visualize social attention weights between the target agent
    and its neighbors.
    
    Args:
        obs_traj:          (obs_len, 2) target observed trajectory
        neighbors_obs:     (N, obs_len, 2) neighbor positions
        neighbor_mask:     (N,) boolean mask
        attention_weights: (N,) attention weights
        save_path:         path to save
        title:             plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_facecolor("#F8F9FA")
    
    if hasattr(obs_traj, "numpy"):
        obs_traj = obs_traj.cpu().numpy()
    if hasattr(neighbors_obs, "numpy"):
        neighbors_obs = neighbors_obs.cpu().numpy()
    if hasattr(neighbor_mask, "numpy"):
        neighbor_mask = neighbor_mask.cpu().numpy()
    if hasattr(attention_weights, "numpy"):
        attention_weights = attention_weights.cpu().numpy()
    
    # Plot target trajectory
    ax.plot(
        obs_traj[:, 0], obs_traj[:, 1],
        color=OBS_COLOR, linewidth=3, label="Target Agent",
        marker="o", markersize=6, zorder=5,
    )
    
    # Plot neighbors with attention-weighted colors
    for i in range(len(neighbor_mask)):
        if not neighbor_mask[i]:
            continue
        
        attn = attention_weights[i]
        color_intensity = min(1.0, attn * 3)
        color = plt.cm.Reds(0.3 + color_intensity * 0.7)
        
        n_traj = neighbors_obs[i]
        ax.plot(
            n_traj[:, 0], n_traj[:, 1],
            color=color, linewidth=1.5 + attn * 3,
            marker="s", markersize=3, alpha=0.7, zorder=3,
        )
        
        # Draw attention line from target to neighbor
        ax.plot(
            [obs_traj[-1, 0], n_traj[-1, 0]],
            [obs_traj[-1, 1], n_traj[-1, 1]],
            color=color, linewidth=attn * 5, alpha=0.4,
            linestyle=":", zorder=2,
        )
        
        # Annotate attention weight
        mid_x = (obs_traj[-1, 0] + n_traj[-1, 0]) / 2
        mid_y = (obs_traj[-1, 1] + n_traj[-1, 1]) / 2
        ax.annotate(
            f"{attn:.2f}",
            (mid_x, mid_y),
            fontsize=9, fontweight="bold",
            color=color, ha="center",
        )
    
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close(fig)
    return fig
