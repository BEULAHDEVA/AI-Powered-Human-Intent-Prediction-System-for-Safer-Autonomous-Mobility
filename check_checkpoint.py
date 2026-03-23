import torch
import sys

ckpt_path = "checkpoints/checkpoint_epoch_70.pt"
try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)

state_dict = ckpt["model_state_dict"]
shape = state_dict.get("trajectory_encoder.input_projection.0.weight").shape
print(f"Input projection shape in epoch 70: {shape}")
