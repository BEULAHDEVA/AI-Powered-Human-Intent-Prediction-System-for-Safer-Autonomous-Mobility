# AI-Powered-Human-Intent-Prediction-System-for-Safer-Autonomous-Mobility

AI system that predicts human intent for safer autonomous driving  Generates multiple future paths with probabilities, capturing real-world uncertainty and behavior. Powered by a Social Transformer, it models interactions between agents to deliver smarter, more reliable trajectory predictions.

##  Overview

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

## Visualization outputs 

<img width="2359" height="2369" alt="scenario_analysis_1" src="https://github.com/user-attachments/assets/14545baa-dda1-4b5c-bb84-2650815089f9" />
<img width="2359" height="2369" alt="scenario_analysis_2" src="https://github.com/user-attachments/assets/e9f1f86d-7328-444a-b1f1-84ebe7bccb30" />
<img width="2359" height="2369" alt="scenario_analysis_3" src="https://github.com/user-attachments/assets/da55fd5c-3f40-4294-94d8-c65a1585c8e4" />
<img width="2359" height="2369" alt="scenario_analysis_4" src="https://github.com/user-attachments/assets/440f9342-ed9d-4cfe-b4d3-3315cea94592" />
<img width="2359" height="2369" alt="scenario_analysis_5" src="https://github.com/user-attachments/assets/9c6f952f-15f7-4886-8fbe-b54c1bba7a5c" />
<img width="2385" height="2369" alt="scenario_analysis_6" src="https://github.com/user-attachments/assets/7527f624-6848-49d3-a541-3666b485f5b9" />
<img width="2359" height="2369" alt="scenario_analysis_7" src="https://github.com/user-attachments/assets/678dac8c-4843-4826-9f53-fe39d748a3b5" />
<img width="2359" height="2369" alt="scenario_analysis_8" src="https://github.com/user-attachments/assets/1fbe0c7c-f4f4-4d25-a0a2-0860a7b358ae" />
<img width="2082" height="730" alt="training_curves" src="https://github.com/user-attachments/assets/cbcdd403-a7c3-42bb-ad46-864518855e79" />


