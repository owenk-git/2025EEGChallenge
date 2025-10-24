"""
Find best C1 and C2 checkpoints from experiments and create submission

Usage:
    python find_best_and_submit.py
"""

import torch
from pathlib import Path
import subprocess

def find_best_checkpoint(checkpoint_dirs, challenge_name):
    """Find the checkpoint with lowest NRMSE across multiple experiment directories"""
    best_nrmse = float('inf')
    best_path = None
    best_info = None

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir) / f"{challenge_name}_aggressive_best.pth"

        if not checkpoint_path.exists():
            continue

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            nrmse = checkpoint['val_nrmse']
            epoch = checkpoint['epoch']

            print(f"  {checkpoint_dir}: NRMSE={nrmse:.4f} (epoch {epoch})")

            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_path = checkpoint_path
                best_info = checkpoint
        except Exception as e:
            print(f"  {checkpoint_dir}: Error loading - {e}")

    return best_path, best_nrmse, best_info


def main():
    print("="*70)
    print("🔍 Finding Best Checkpoints from Experiments")
    print("="*70)

    # C1 experiment directories
    c1_dirs = [
        'checkpoints_c1_huber_median',
        'checkpoints_c1_huber_trimmed',
        'checkpoints_c1_mae_wide',
        'checkpoints_aggressive_c1'  # default
    ]

    # C2 experiment directories
    c2_dirs = [
        'checkpoints_c2_extreme',
        'checkpoints_c2_mae',
        'checkpoints_aggressive_c2'  # default
    ]

    print("\n📊 Challenge 1 Results:")
    print("-" * 70)
    c1_best_path, c1_best_nrmse, c1_info = find_best_checkpoint(c1_dirs, 'c1')

    print("\n📊 Challenge 2 Results:")
    print("-" * 70)
    c2_best_path, c2_best_nrmse, c2_info = find_best_checkpoint(c2_dirs, 'c2')

    print("\n" + "="*70)
    print("🏆 BEST MODELS")
    print("="*70)

    if c1_best_path:
        print(f"C1: {c1_best_path}")
        print(f"    NRMSE: {c1_best_nrmse:.4f} (Target: 0.90)")
        print(f"    Epoch: {c1_info['epoch']}")
    else:
        print("C1: No checkpoint found!")

    if c2_best_path:
        print(f"C2: {c2_best_path}")
        print(f"    NRMSE: {c2_best_nrmse:.4f} (Target: 0.83)")
        print(f"    Epoch: {c2_info['epoch']}")
    else:
        print("C2: No checkpoint found!")

    # Calculate expected overall score
    if c1_best_nrmse < float('inf') and c2_best_nrmse < float('inf'):
        overall = 0.3 * c1_best_nrmse + 0.7 * c2_best_nrmse
        print(f"\nExpected Overall: {overall:.4f}")
        print(f"Current Best: 1.11")
        if overall < 1.11:
            print(f"🎉 IMPROVEMENT: {1.11 - overall:.4f}")
        else:
            print(f"⚠️  Not better than current best")

        # SOTA comparison
        print(f"\nSOTA Target: 0.978")
        if overall < 0.978:
            print(f"🏆 BEATS SOTA by {0.978 - overall:.4f}!")
        else:
            print(f"⚠️  Gap to SOTA: {overall - 0.978:.4f}")

    # Create submission if models found
    if c1_best_path and c2_best_path:
        print("\n" + "="*70)
        print("📦 Creating Submission")
        print("="*70)

        # Copy to standard checkpoint names
        import shutil
        checkpoint_dir = Path('checkpoints_best_aggressive')
        checkpoint_dir.mkdir(exist_ok=True)

        c1_dest = checkpoint_dir / 'c1_best.pth'
        c2_dest = checkpoint_dir / 'c2_best.pth'

        shutil.copy(c1_best_path, c1_dest)
        shutil.copy(c2_best_path, c2_dest)

        print(f"✅ Copied best checkpoints to {checkpoint_dir}")

        # Create submission
        print("\n🚀 Creating submission ZIP...")
        result = subprocess.run([
            'python', 'create_submission.py',
            '--model_c1', str(c1_dest),
            '--model_c2', str(c2_dest),
            '--output', 'aggressive_best.zip'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Submission created: aggressive_best.zip")
            print("\n" + "="*70)
            print("🎯 READY TO SUBMIT!")
            print("="*70)
            print("Upload aggressive_best.zip to Codabench")
            print(f"Expected score: {overall:.4f}")
        else:
            print(f"❌ Error creating submission:")
            print(result.stderr)
    else:
        print("\n⚠️  Cannot create submission - missing checkpoints")

    print("="*70)


if __name__ == "__main__":
    main()
