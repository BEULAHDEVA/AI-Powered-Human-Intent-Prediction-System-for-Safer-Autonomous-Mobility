"""Quick sanity check for the entire pipeline."""
from data.nuscenes_dataset import NuScenesTrajectoryDataset, create_dataloaders
from config import Config
from models.social_transformer import SocialTransformer
from utils.losses import TrajectoryPredictionLoss
from utils.metrics import compute_all_metrics
import torch

cfg = Config()
cfg.ensure_dirs()
cfg.NUM_WORKERS = 0  # for quick test

# Test dataset
print("=" * 50)
print("1. Testing Data Pipeline")
print("=" * 50)
ds = NuScenesTrajectoryDataset("train", cfg, use_synthetic=True)
print(f"   Train samples: {len(ds)}")

sample = ds[0]
for key, val in sample.items():
    print(f"   {key}: {val.shape}")

# Test dataloader
print("\n2. Testing DataLoader")
train_loader, val_loader, test_loader = create_dataloaders(cfg, use_synthetic=True)
batch = next(iter(train_loader))
print(f"   Batch obs_traj: {batch['obs_traj'].shape}")
print(f"   Batch pred_traj: {batch['pred_traj'].shape}")

# Test model forward pass
print("\n3. Testing Model Forward Pass")
model = SocialTransformer(cfg)
predictions, mode_probs, goals, social_attn = model(
    batch["obs_traj"], batch["obs_vel"], batch["obs_acc"],
    batch["obs_heading"], batch["obs_context"],
    batch["neighbors_obs"], batch["neighbor_mask"]
)
print(f"   Predictions: {predictions.shape}  (batch, K={cfg.NUM_MODES}, pred_len={cfg.PRED_LEN}, 2)")
print(f"   Mode probs:  {mode_probs.shape}  (batch, K={cfg.NUM_MODES})")
print(f"   Goals:        {goals.shape}  (batch, K={cfg.NUM_MODES}, 2)")
print(f"   Social attn:  {social_attn.shape}  (batch, max_neighbors={cfg.MAX_NEIGHBORS})")

# Test loss
print("\n4. Testing Loss Computation")
loss_fn = TrajectoryPredictionLoss(cfg)
total_loss, loss_dict = loss_fn(predictions, mode_probs, batch["pred_traj"])
for key, val in loss_dict.items():
    print(f"   {key}: {val:.4f}")

# Test metrics
print("\n5. Testing Metrics")
metrics = compute_all_metrics(predictions, mode_probs, batch["pred_traj"])
for key, val in metrics.items():
    print(f"   {key}: {val:.4f}")

# Test backward pass
print("\n6. Testing Backward Pass")
total_loss.backward()
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms.append(param.grad.norm().item())
print(f"   Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")
print(f"   Parameters with gradients: {len(grad_norms)}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
