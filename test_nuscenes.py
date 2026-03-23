"""Quick test: load real nuScenes data and extract trajectories."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.nuscenes_dataset import NuScenesTrajectoryDataset

cfg = Config()
cfg.ensure_dirs()
cfg.NUM_WORKERS = 0

print("=" * 60)
print("Testing nuScenes Data Extraction")
print("=" * 60)
print(f"  Dataset root: {cfg.NUSCENES_DATAROOT}")
print(f"  Version: {cfg.NUSCENES_VERSION}")
print(f"  Obs steps: {cfg.OBS_LEN}, Pred steps: {cfg.PRED_LEN}")
print()

# Load train split from real nuScenes
train_ds = NuScenesTrajectoryDataset("train", cfg, use_synthetic=False)
print(f"\nTrain samples extracted: {len(train_ds)}")

if len(train_ds) > 0:
    sample = train_ds[0]
    print("\nSample shapes:")
    for key, val in sample.items():
        print(f"  {key}: {val.shape}")
    
    val_ds = NuScenesTrajectoryDataset("val", cfg, use_synthetic=False)
    test_ds = NuScenesTrajectoryDataset("test", cfg, use_synthetic=False)
    print(f"\nVal samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Total: {len(train_ds) + len(val_ds) + len(test_ds)}")
    print("\n✅ nuScenes data extraction successful!")
else:
    print("\n⚠️  No trajectory samples extracted. Checking categories...")
    # Debug: check what categories exist
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg.NUSCENES_VERSION, dataroot=cfg.NUSCENES_DATAROOT, verbose=False)
    cats = set()
    for scene in nusc.scene:
        sample_token = scene["first_sample_token"]
        sample = nusc.get("sample", sample_token)
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cats.add(ann["category_name"])
    print("  Available categories:")
    for c in sorted(cats):
        print(f"    - {c}")
