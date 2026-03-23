"""
Trajectory Encoder — Transformer-Based
========================================
Encodes observed trajectory history (positions + velocities) into
a rich context embedding using multi-head self-attention.

Architecture:
  Input Projection → Positional Encoding → Transformer Encoder → Context Vector
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal awareness.
    Adds time-step information to the input embeddings.
    """
    
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrajectoryEncoder(nn.Module):
    """
    Transformer-based encoder for trajectory sequences.
    
    Takes (x, y, vx, vy) per timestep and produces a context embedding
    that captures:
    - Temporal motion patterns (acceleration, direction changes)
    - Implicit intent signals (slowing down, veering)
    """
    
    def __init__(
        self,
        input_dim=4,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        ff_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Project raw (x, y, vx, vy) into embedding space
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Temporal positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Layer norm for final output
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, obs_traj, obs_vel, obs_acc, obs_heading, obs_context):
        """
        Encode observed trajectory.
        
        Args:
            obs_traj: (batch, obs_len, 2) normalized positions
            obs_vel:  (batch, obs_len, 2) velocities
            obs_acc:  (batch, obs_len, 2) accelerations
            obs_heading: (batch, obs_len, 1) yaw heading
            obs_context: (batch, obs_len, 1) map flags
        
        Returns:
            context:    (batch, embed_dim) trajectory context vector
            seq_output: (batch, obs_len, embed_dim) per-timestep features
        """
        # Concatenate all features → (batch, obs_len, 8)
        x = torch.cat([obs_traj, obs_vel, obs_acc, obs_heading, obs_context], dim=-1)
        
        # Project to embedding space
        x = self.input_projection(x)   # (batch, obs_len, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        seq_output = self.transformer_encoder(x)  # (batch, obs_len, embed_dim)
        seq_output = self.layer_norm(seq_output)
        
        # Use last timestep as the context vector (captures full history)
        context = seq_output[:, -1, :]  # (batch, embed_dim)
        
        return context, seq_output
