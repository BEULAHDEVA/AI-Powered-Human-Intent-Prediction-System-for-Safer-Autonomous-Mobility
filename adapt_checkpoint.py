import torch
import os

def adapt_checkpoint(checkpoint_path, output_path):
    print(f"Adapting {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    opt_state = checkpoint["optimizer_state_dict"]
    
    # ── Identify target parameter ──
    # We need to find the index of 'trajectory_encoder.input_projection.0.weight'
    # in the model to find it in the optimizer.
    # However, since we don't have the model object here, we can try to match shapes.
    
    target_key = "trajectory_encoder.input_projection.0.weight"
    if target_key in state_dict:
        old_weight = state_dict[target_key]
        print(f"  Modifying {target_key}: {old_weight.shape} -> (64, 8)")
        
        new_weight = torch.zeros((64, 8))
        new_weight[:, :4] = old_weight
        state_dict[target_key] = new_weight
        
        # ── Adapt Optimizer State ──
        # Find which state entry corresponds to this weight
        # In Adam, each state[param_id] is a dict with 'exp_avg', 'exp_avg_sq', etc.
        # We can look for tensors with shape (64, 4) in the optimizer state.
        
        adapted_opt_count = 0
        for pid, state in opt_state["state"].items():
            for s_key, s_val in state.items():
                if torch.is_tensor(s_val) and s_val.shape == torch.Size([64, 4]):
                    print(f"    Adapting optimizer buffer '{s_key}' for param ID {pid}")
                    new_s_val = torch.zeros((64, 8))
                    new_s_val[:, :4] = s_val
                    state[s_key] = new_s_val
                    adapted_opt_count += 1
        
        print(f"  Adapted {adapted_opt_count} optimizer buffers.")
    else:
        print(f"  Error: {target_key} not found.")

    torch.save(checkpoint, output_path)
    print(f"Done! Saved adapted checkpoint to {output_path}")

if __name__ == "__main__":
    ckpt_path = "checkpoints/checkpoint_epoch_20.pt"
    out_path = "checkpoints/checkpoint_epoch_20_adapted.pt"
    if os.path.exists(ckpt_path):
        adapt_checkpoint(ckpt_path, out_path)
    else:
        print("Source checkpoint not found.")
