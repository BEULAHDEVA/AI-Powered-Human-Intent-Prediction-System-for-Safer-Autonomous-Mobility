"""
Multi-Modal Trajectory Decoder
================================
Generates K diverse trajectory predictions, each with an associated
probability score. Uses goal-conditioned residual connections for
physically plausible predictions.

Key design choices:
- K=3 modes: captures primary intent + alternatives
- Residual prediction: outputs offsets from linear extrapolation
- Mode probability: softmax over learned mode logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalConditionedDecoder(nn.Module):
    """
    Single-mode trajectory decoder with goal conditioning.
    
    Predicts a complete future trajectory by:
    1. Predicting a goal (endpoint)
    2. Filling in intermediate waypoints conditioned on the goal
    3. Adding residual corrections from linear extrapolation
    """
    
    def __init__(self, context_dim, pred_len, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        self.pred_len = pred_len
        self.context_dim = context_dim
        
        # Goal predictor: predicts final position (x, y)
        self.goal_predictor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # goal (x, y)
        )
        
        # Waypoint generator: fills between start and goal
        self.waypoint_generator = nn.Sequential(
            nn.Linear(context_dim + 2, hidden_dim),  # +2 for goal
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),  # all waypoints (x, y)
        )
    
    def forward(self, context, last_obs_vel):
        """
        Predict a single trajectory mode.
        
        Args:
            context:      (batch, context_dim) fused context
            last_obs_vel: (batch, 2) velocity at last observed step
        
        Returns:
            trajectory: (batch, pred_len, 2) predicted positions
            goal:       (batch, 2) predicted goal position
        """
        batch = context.shape[0]
        
        # Predict goal
        goal = self.goal_predictor(context)  # (batch, 2)
        
        # Generate waypoints conditioned on context + goal
        decoder_input = torch.cat([context, goal], dim=-1)
        waypoints = self.waypoint_generator(decoder_input)
        waypoints = waypoints.view(batch, self.pred_len, 2)
        
        # Add linear extrapolation as residual baseline
        # This provides a strong inductive bias for smooth motion
        dt = torch.arange(1, self.pred_len + 1, device=context.device).float()
        dt = dt.unsqueeze(0).unsqueeze(-1)  # (1, pred_len, 1)
        linear_extrapolation = last_obs_vel.unsqueeze(1) * dt * 0.5  # scale by dt
        
        trajectory = waypoints + linear_extrapolation
        
        return trajectory, goal


class MultiModalDecoder(nn.Module):
    """
    Multi-Modal Trajectory Decoder.
    
    Generates K trajectory hypotheses with associated probabilities.
    Each mode has its own GoalConditionedDecoder to encourage diversity.
    
    For K=3:
    - Mode 0: Most likely trajectory (e.g., continue straight)
    - Mode 1: Alternative 1 (e.g., turn left)
    - Mode 2: Alternative 2 (e.g., turn right / stop)
    """
    
    def __init__(
        self,
        context_dim,
        pred_len,
        num_modes=3,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        
        self.num_modes = num_modes
        self.pred_len = pred_len
        
        # K independent decoders (one per mode)
        self.mode_decoders = nn.ModuleList([
            GoalConditionedDecoder(context_dim, pred_len, hidden_dim, dropout)
            for _ in range(num_modes)
        ])
        
        # Mode probability predictor
        self.mode_classifier = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modes),
        )
    
    def forward(self, context, last_obs_vel):
        """
        Generate multi-modal trajectory predictions.
        
        Args:
            context:      (batch, context_dim) fused trajectory + social context
            last_obs_vel: (batch, 2) velocity at last observed step
        
        Returns:
            predictions:   (batch, K, pred_len, 2) K trajectory predictions
            mode_probs:    (batch, K) probability for each mode (sums to 1)
            goals:         (batch, K, 2) predicted goal for each mode
        """
        batch = context.shape[0]
        
        all_trajectories = []
        all_goals = []
        
        for decoder in self.mode_decoders:
            traj, goal = decoder(context, last_obs_vel)
            all_trajectories.append(traj)
            all_goals.append(goal)
        
        # Stack modes: (batch, K, pred_len, 2)
        predictions = torch.stack(all_trajectories, dim=1)
        goals = torch.stack(all_goals, dim=1)  # (batch, K, 2)
        
        # Mode probabilities
        mode_logits = self.mode_classifier(context)  # (batch, K)
        mode_probs = F.softmax(mode_logits, dim=-1)
        
        return predictions, mode_probs, goals
