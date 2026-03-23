"""
SocialTransformer — End-to-End Model
======================================
Combines the Trajectory Encoder, Social Pooling, and Multi-Modal
Decoder into a single model for intent and trajectory prediction.

  ┌──────────────────────────────┐
  │   Observed Trajectory        │
  │   (x, y, vx, vy) × T_obs    │
  └────────────┬─────────────────┘
               ▼
  ┌────────────────────────────┐
  │  Trajectory Encoder        │
  │  (Transformer)             │
  └────────────┬───────────────┘
               │
  ┌────────────┴───────────────┐
  │        Fusion Layer        │◄── Social Pooling ◄── Neighbor Trajectories
  └────────────┬───────────────┘
               ▼
  ┌────────────────────────────┐
  │ Multi-Modal Decoder (K=3)  │
  │ Goal-Conditioned Residual  │
  └────────────┬───────────────┘
               ▼
  ┌────────────────────────────┐
  │ K Trajectory Predictions   │
  │ + Mode Probabilities       │
  └────────────────────────────┘
"""

import torch
import torch.nn as nn

from .trajectory_encoder import TrajectoryEncoder
from .social_pooling import SocialPooling
from .multimodal_decoder import MultiModalDecoder


class SocialTransformer(nn.Module):
    """
    Social-Aware Multi-Modal Trajectory Prediction Model.
    
    Inputs:
        - Observed trajectory (x, y) for T_obs timesteps
        - Observed velocities (vx, vy) for T_obs timesteps
        - Neighbor trajectories for social context
    
    Outputs:
        - K predicted future trajectories (x, y) for T_pred timesteps
        - Mode probabilities for each prediction
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # ── Trajectory Encoder ──
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=config.INPUT_DIM,
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_ENCODER_LAYERS,
            ff_dim=config.FF_DIM,
            dropout=config.DROPOUT,
        )
        
        # ── Social Pooling ──
        self.social_pooling = SocialPooling(
            embed_dim=config.EMBED_DIM,
            social_embed_dim=config.SOCIAL_EMBED_DIM,
            num_heads=config.NUM_HEADS,
            max_neighbors=config.MAX_NEIGHBORS,
            social_radius=config.SOCIAL_RADIUS,
            dropout=config.DROPOUT,
        )
        
        # ── Fusion Layer ──
        # Combine trajectory context + social context
        fusion_input_dim = config.EMBED_DIM + config.SOCIAL_EMBED_DIM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.EMBED_DIM * 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM),
            nn.LayerNorm(config.EMBED_DIM),
        )
        
        # ── Multi-Modal Decoder ──
        self.decoder = MultiModalDecoder(
            context_dim=config.EMBED_DIM,
            pred_len=config.PRED_LEN,
            num_modes=config.NUM_MODES,
            hidden_dim=config.FF_DIM,
            dropout=config.DROPOUT,
        )
    
    def forward(self, obs_traj, obs_vel, obs_acc, obs_heading, obs_context, neighbors_obs, neighbor_mask):
        """
        Full forward pass.
        
        Args:
            obs_traj:      (batch, obs_len, 2) observed positions
            obs_vel:       (batch, obs_len, 2) observed velocities
            obs_acc:       (batch, obs_len, 2) accelerations
            obs_heading:   (batch, obs_len, 1) yaw heading
            obs_context:   (batch, obs_len, 1) map contexts
            neighbors_obs: (batch, max_N, obs_len, 2) neighbor positions
            neighbor_mask: (batch, max_N) boolean mask for valid neighbors
        
        Returns:
            predictions: (batch, K, pred_len, 2) predicted trajectories
            mode_probs:  (batch, K) mode probabilities
            goals:       (batch, K, 2) predicted endpoints
            social_attn: (batch, max_N) social attention weights
        """
        # 1. Encode target trajectory
        agent_context, seq_features = self.trajectory_encoder(
            obs_traj, obs_vel, obs_acc, obs_heading, obs_context
        )
        # agent_context: (batch, embed_dim)
        
        # 2. Compute social context
        social_context, social_attn = self.social_pooling(
            agent_context, neighbors_obs, neighbor_mask
        )
        # social_context: (batch, social_embed_dim)
        
        # 3. Fuse trajectory + social context
        fused = torch.cat([agent_context, social_context], dim=-1)
        fused = self.fusion(fused)  # (batch, embed_dim)
        
        # 4. Decode multi-modal predictions
        last_obs_vel = obs_vel[:, -1, :]  # (batch, 2)
        predictions, mode_probs, goals = self.decoder(fused, last_obs_vel)
        
        return predictions, mode_probs, goals, social_attn
    
    def predict(self, obs_traj, obs_vel, obs_acc, obs_heading, obs_context, neighbors_obs, neighbor_mask):
        """
        Inference-only prediction (no gradients).
        
        Returns the best trajectory (highest probability mode).
        """
        self.eval()
        with torch.no_grad():
            predictions, mode_probs, goals, social_attn = self.forward(
                obs_traj, obs_vel, obs_acc, obs_heading, obs_context, 
                neighbors_obs, neighbor_mask
            )
        
        # Select best mode per sample
        best_mode_idx = mode_probs.argmax(dim=1)  # (batch,)
        batch_idx = torch.arange(predictions.shape[0], device=predictions.device)
        best_predictions = predictions[batch_idx, best_mode_idx]  # (batch, pred_len, 2)
        
        return best_predictions, predictions, mode_probs, social_attn
    
    def get_num_params(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
