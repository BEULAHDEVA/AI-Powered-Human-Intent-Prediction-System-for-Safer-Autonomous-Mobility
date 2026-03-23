"""
Social Pooling Module
======================
Attention-based social context aggregation that models how
pedestrians influence each other's trajectories.

Key insight: Pedestrians and cyclists don't move in isolation.
They avoid collisions, follow social norms (walking on the right),
and adjust speed/direction based on surrounding agents.

This module uses cross-attention between the target agent and its
neighbors, weighted by spatial proximity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SocialPooling(nn.Module):
    """
    Attention-based Social Pooling.
    
    Given a target agent's trajectory encoding and neighbor trajectories,
    produces a social context vector that captures:
    - Which neighbors are most influential
    - How their motion affects the target agent
    - Implicit collision-avoidance signals
    
    Architecture:
      Neighbor Encoding → Distance-Weighted Cross-Attention → Social Context
    """
    
    def __init__(
        self,
        embed_dim=64,
        social_embed_dim=64,
        num_heads=4,
        max_neighbors=20,
        social_radius=10.0,
        dropout=0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.social_embed_dim = social_embed_dim
        self.num_heads = num_heads
        self.social_radius = social_radius
        
        # Encode neighbor trajectories (positions over time)
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(2, embed_dim),  # per-timestep position → embedding
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Temporal aggregation for each neighbor (simple attention over time)
        self.temporal_attention = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )
        
        # Cross-attention: target queries, neighbor keys/values
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Distance-based attention bias
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // num_heads),
            nn.ReLU(),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, social_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(social_embed_dim, social_embed_dim),
        )
        
        self.layer_norm = nn.LayerNorm(social_embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _encode_neighbors(self, neighbors_obs, neighbor_mask):
        """
        Encode each neighbor's observed trajectory.
        
        Args:
            neighbors_obs: (batch, max_N, obs_len, 2) neighbor positions
            neighbor_mask:  (batch, max_N) boolean mask
        
        Returns:
            neighbor_embeddings: (batch, max_N, embed_dim)
            distances: (batch, max_N, 1) distance to each neighbor
        """
        batch, max_n, obs_len, _ = neighbors_obs.shape
        
        # Encode all timesteps: (batch, max_N, obs_len, 2) → (batch, max_N, obs_len, embed_dim)
        n_embed = self.neighbor_encoder(neighbors_obs)
        
        # Temporal attention to aggregate over time
        # scores: (batch, max_N, obs_len, 1)
        time_scores = self.temporal_attention(n_embed)
        
        # Mask invalid neighbors
        mask_expanded = neighbor_mask.unsqueeze(-1).unsqueeze(-1).float()  # (batch, max_N, 1, 1)
        time_scores = time_scores * mask_expanded + (-1e9) * (1 - mask_expanded)
        
        time_weights = F.softmax(time_scores, dim=2)  # over obs_len dim
        
        # Weighted sum: (batch, max_N, embed_dim)
        neighbor_embeddings = (n_embed * time_weights).sum(dim=2)
        
        # Compute distances (use last observed position of each neighbor)
        last_pos = neighbors_obs[:, :, -1, :]  # (batch, max_N, 2)
        distances = torch.norm(last_pos, dim=-1, keepdim=True)  # (batch, max_N, 1)
        
        return neighbor_embeddings, distances
    
    def forward(self, agent_context, neighbors_obs, neighbor_mask):
        """
        Compute social context vector.
        
        Args:
            agent_context:  (batch, embed_dim) target agent's trajectory encoding
            neighbors_obs:  (batch, max_N, obs_len, 2) neighbor positions (relative)
            neighbor_mask:   (batch, max_N) boolean mask for valid neighbors
        
        Returns:
            social_context: (batch, social_embed_dim) social context vector
            attention_weights: (batch, max_N) attention weights for visualization
        """
        batch = agent_context.shape[0]
        max_n = neighbors_obs.shape[1]
        
        # Handle case with no neighbors
        if not neighbor_mask.any():
            return (
                torch.zeros(batch, self.social_embed_dim, device=agent_context.device),
                torch.zeros(batch, max_n, device=agent_context.device),
            )
        
        # Encode neighbors
        neighbor_embeddings, distances = self._encode_neighbors(
            neighbors_obs, neighbor_mask
        )
        
        # ── Cross-Attention ──
        # Query from target agent, keys/values from neighbors
        Q = self.query_proj(agent_context).unsqueeze(1)  # (batch, 1, embed_dim)
        K = self.key_proj(neighbor_embeddings)             # (batch, max_N, embed_dim)
        V = self.value_proj(neighbor_embeddings)           # (batch, max_N, embed_dim)
        
        # Scaled dot-product attention
        d_k = self.embed_dim // self.num_heads
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        # (batch, 1, max_N)
        
        # Distance-based bias: closer neighbors get more attention
        dist_bias = self.distance_embedding(
            1.0 / (distances + 1e-6)
        )  # (batch, max_N, embed_dim // num_heads)
        dist_bias = dist_bias.mean(dim=-1, keepdim=True).transpose(1, 2)
        # (batch, 1, max_N)
        attn_scores = attn_scores + dist_bias
        
        # Mask invalid neighbors
        attn_mask = (~neighbor_mask).unsqueeze(1)  # (batch, 1, max_N)
        attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        
        # Check if all neighbors are masked (prevent NaN from softmax)
        all_masked = (~neighbor_mask).all(dim=1)  # (batch,)
        
        # For fully-masked rows, set scores to 0 so softmax gives uniform dist
        # (we'll zero out the result afterwards via isolation_mask)
        all_masked_expanded = all_masked.view(-1, 1, 1).expand_as(attn_scores)
        attn_scores = torch.where(all_masked_expanded, torch.zeros_like(attn_scores), attn_scores)
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, max_N)
        # Safety net for any remaining NaN
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregated social context
        social_raw = torch.bmm(attn_weights, V).squeeze(1)  # (batch, embed_dim)
        
        # Zero out for fully isolated agents (non-in-place to preserve gradients)
        isolation_mask = all_masked.unsqueeze(-1).float()  # (batch, 1)
        social_raw = social_raw * (1.0 - isolation_mask)
        
        # Project to social embedding
        social_context = self.output_proj(social_raw)
        social_context = self.layer_norm(social_context)
        
        # Return attention weights for visualization (detached, safe for in-place)
        attn_vis = attn_weights.squeeze(1).detach().clone()  # (batch, max_N)
        attn_vis[all_masked] = 0.0
        
        return social_context, attn_vis
