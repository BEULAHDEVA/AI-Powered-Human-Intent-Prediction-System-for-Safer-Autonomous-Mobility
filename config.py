"""
Configuration for Social-Aware Multi-Modal Trajectory Prediction
================================================================
All hyperparameters and settings in one place.
"""

import os


class Config:
    """Central configuration class."""
    
    # ── Paths ──────────────────────────────────────────────────
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    
    # ── Dataset ────────────────────────────────────────────────
    # nuScenes dataset version: 'v1.0-mini' for quick dev, 'v1.0-trainval' for full
    NUSCENES_VERSION = "v1.0-mini"
    NUSCENES_DATAROOT = os.path.join(DATA_DIR, "nuscenes")
    
    # Categories to track
    PEDESTRIAN_CATEGORIES = [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.police_officer",
    ]
    CYCLIST_CATEGORIES = [
        "vehicle.bicycle",
        "vehicle.motorcycle",
    ]
    TARGET_CATEGORIES = PEDESTRIAN_CATEGORIES + CYCLIST_CATEGORIES
    
    # ── Temporal Settings ──────────────────────────────────────
    SAMPLE_RATE_HZ = 2          # nuScenes annotation frequency
    OBS_SECONDS = 2             # seconds of observed history
    PRED_SECONDS = 3            # seconds to predict into the future
    OBS_LEN = OBS_SECONDS * SAMPLE_RATE_HZ      # 4 timesteps
    PRED_LEN = PRED_SECONDS * SAMPLE_RATE_HZ    # 6 timesteps
    SEQ_LEN = OBS_LEN + PRED_LEN                # 10 timesteps total
    
    # ── Model Architecture ─────────────────────────────────────
    INPUT_DIM = 8               # (x, y, vx, vy, ax, ay, heading, crosswalk_flag)
    EMBED_DIM = 64              # embedding dimension
    NUM_HEADS = 4               # transformer attention heads
    NUM_ENCODER_LAYERS = 3      # transformer encoder depth
    NUM_DECODER_LAYERS = 3      # decoder depth
    DROPOUT = 0.2               # dropout rate increased for stability
    FF_DIM = 128                # feed-forward hidden dimension
    
    # Social Pooling
    SOCIAL_RADIUS = 10.0        # meters — neighbor consideration radius
    MAX_NEIGHBORS = 20          # max neighbors for social pooling
    SOCIAL_EMBED_DIM = 64       # social context embedding dimension
    
    # Multi-Modal Prediction
    NUM_MODES = 3               # number of trajectory hypotheses (K)
    
    # ── Training ───────────────────────────────────────────────
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4        # Lowered learning rate for stability
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 150            # Train longer for competition
    WARMUP_EPOCHS = 5
    VARIETY_LOSS_WEIGHT = 5.0   # Significantly increase to force modes to spread out and avoid collapse
    MODE_LOSS_WEIGHT = 0.1      # Decrease to prevent probability exploding early on
    GRAD_CLIP = 1.0             # gradient clipping norm
    
    # ── Evaluation ─────────────────────────────────────────────
    EVAL_BATCH_SIZE = 128
    
    # ── Device ─────────────────────────────────────────────────
    DEVICE = "cuda"  # will fallback to cpu if not available
    
    # ── Misc ───────────────────────────────────────────────────
    SEED = 42
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories."""
        for d in [cls.PROCESSED_DIR, cls.CHECKPOINT_DIR, cls.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)
