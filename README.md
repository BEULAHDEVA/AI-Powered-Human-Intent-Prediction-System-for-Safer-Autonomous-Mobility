# AI-Powered-Human-Intent-Prediction-System-for-Safer-Autonomous-Mobility

AI system that predicts human intent for safer autonomous driving 🚗✨ Generates multiple future paths with probabilities, capturing real-world uncertainty and behavior. Powered by a Social Transformer, it models interactions between agents to deliver smarter, more reliable trajectory predictions.

## 🎯 Overview

A **SocialTransformer** model for predicting future trajectories (3 seconds) of pedestrians and cyclists from 2 seconds of observed motion history, designed for **L4 urban autonomous driving**.

## Architecture

```
Observed Trajectory (x,y,vx,vy) × 4 steps
        │
        ▼
┌─────────────────────────────┐
│  Trajectory Encoder         │  Transformer with positional encoding
│  (Multi-Head Self-Attention)│  + GELU activation
└───────────┬─────────────────┘
            │
            ├──── Fusion ◄──── Social Pooling ◄──── Neighbor Trajectories
            │                   (Cross-Attention)     (within 10m radius)
            ▼
┌─────────────────────────────┐
│  Multi-Modal Decoder (K=3)  │  Goal-conditioned + residual
│  + Mode Probability Head    │  
└───────────┬─────────────────┘
            ▼
  3 Predicted Trajectories (6 steps each)
  + Probability per mode
```

## Key Features

- **Transformer Encoder**: Captures temporal motion patterns with self-attention
- **Social Pooling**: Attention-based neighbor influence modeling with distance weighting
- **Multi-Modal Output**: K=3 diverse trajectory hypotheses with probabilities
- **Goal-Conditioned Decoding**: Predicts endpoint first, then fills waypoints
- **Best-of-N Training**: Winner-takes-all loss for multi-modal learning
- **Variety Loss**: Prevents mode collapse, encourages diverse predictions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train with Synthetic Data (for development)

```bash
python train.py --epochs 50
```

### 3. Train with nuScenes Dataset

First, download the [nuScenes dataset](https://www.nuscenes.org/download) and place it in `data/raw/nuscenes/`.

```bash
python train.py --nuscenes --epochs 100
```

### 4. Evaluate

```bash
python evaluate.py
```

## Project Structure

```
mahe/
├── config.py                    # All hyperparameters and settings
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Dependencies
│
├── models/
│   ├── trajectory_encoder.py    # Transformer-based motion encoder
│   ├── social_pooling.py        # Attention-based social context
│   ├── multimodal_decoder.py    # K=3 goal-conditioned decoder
│   └── social_transformer.py    # End-to-end model
│
├── data/
│   ├── nuscenes_dataset.py      # nuScenes data loading + synthetic data
│   └── preprocessing.py         # Augmentation and coordinate transforms
│
├── utils/
│   ├── losses.py                # Best-of-N, variety, mode probability losses
│   ├── metrics.py               # ADE, FDE, minADE, minFDE metrics
│   └── visualization.py         # Trajectory plotting utilities
│
├── checkpoints/                 # Saved model weights
└── results/                     # Evaluation outputs and plots
```

## Metrics

| Metric | Description |
|--------|-------------|
| **ADE** | Mean Euclidean distance between predicted and GT across all timesteps |
| **FDE** | Euclidean distance at the final predicted timestep |
| **minADE** | Best ADE across K modes (standard multi-modal metric) |
| **minFDE** | Best FDE across K modes |
| **ML-ADE** | ADE of the most likely mode (highest probability) |
| **ML-FDE** | FDE of the most likely mode |

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `OBS_SECONDS` | 2 | Observation window |
| `PRED_SECONDS` | 3 | Prediction horizon |
| `SAMPLE_RATE_HZ` | 2 | nuScenes annotation frequency |
| `NUM_MODES` | 3 | Number of trajectory hypotheses |
| `EMBED_DIM` | 64 | Embedding dimension |
| `NUM_HEADS` | 4 | Attention heads |
| `SOCIAL_RADIUS` | 10.0m | Neighbor consideration radius |
| `LEARNING_RATE` | 1e-3 | Initial learning rate |

## Dataset

This project is designed for the **nuScenes** dataset. The data pipeline:

1. Extracts pedestrian/cyclist annotations from nuScenes scenes
2. Builds sliding-window sequences (2s observed + 3s future)
3. Computes velocities via finite differences
4. Extracts neighbor trajectories within 10m social radius
5. Normalizes coordinates relative to last observed position

For development without nuScenes, synthetic data with realistic motion patterns is generated automatically.
