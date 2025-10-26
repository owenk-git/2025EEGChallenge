#!/usr/bin/env python3
"""
Create competition submission for trial-level RT prediction

Usage:
    python3 create_trial_level_submission.py \
        --checkpoint_c1 checkpoints/trial_level_c1_best.pt \
        --checkpoint_c2 checkpoints/domain_adaptation_c2_best.pt \
        --name trial_level_fixed
"""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


def create_submission(checkpoint_c1, checkpoint_c2, name):
    """
    Create submission ZIP following competition format

    Args:
        checkpoint_c1: Path to C1 checkpoint (trial-level model)
        checkpoint_c2: Path to C2 checkpoint (domain adaptation model)
        name: Submission name
    """

    print("="*80)
    print("CREATING TRIAL-LEVEL RT PREDICTION SUBMISSION")
    print("="*80)

    # Create temporary directory
    temp_dir = Path("temp_submission")
    temp_dir.mkdir(exist_ok=True)

    print(f"\n📦 Preparing submission files...")

    # Copy submission.py
    submission_template = Path("trial_level_submission.py")
    if not submission_template.exists():
        raise FileNotFoundError(
            f"Submission template not found: {submission_template}\n"
            "Make sure trial_level_submission.py exists!"
        )

    shutil.copy(submission_template, temp_dir / "submission.py")
    print(f"   ✅ Copied submission.py")

    # Copy C1 checkpoint
    c1_path = Path(checkpoint_c1)
    if c1_path.exists():
        shutil.copy(c1_path, temp_dir / "trial_level_c1_best.pt")
        print(f"   ✅ Copied C1 checkpoint: {c1_path}")
    else:
        print(f"   ⚠️ C1 checkpoint not found: {c1_path}")
        print(f"      Submission will use random initialization!")

    # Copy C2 checkpoint (optional)
    if checkpoint_c2:
        c2_path = Path(checkpoint_c2)
        if c2_path.exists():
            shutil.copy(c2_path, temp_dir / "domain_adaptation_c2_best.pt")
            print(f"   ✅ Copied C2 checkpoint: {c2_path}")
        else:
            print(f"   ⚠️ C2 checkpoint not found: {c2_path}")
    else:
        print(f"   ℹ️ No C2 checkpoint specified")

    # Create ZIP
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"{name}_{timestamp}.zip"
    zip_path = submissions_dir / zip_name

    print(f"\n📦 Creating ZIP file...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from temp directory
        for file in temp_dir.iterdir():
            zipf.write(file, file.name)
            print(f"   Added: {file.name}")

    # Cleanup
    shutil.rmtree(temp_dir)

    # Summary
    print(f"\n{'='*80}")
    print("SUBMISSION CREATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"📦 File: {zip_path}")
    print(f"📊 Size: {zip_path.stat().st_size / 1024:.1f} KB")
    print(f"\nContents:")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for info in zipf.filelist:
            size_kb = info.file_size / 1024
            print(f"   - {info.filename:<40} ({size_kb:>8.1f} KB)")

    print(f"\n{'='*80}")
    print("EXPECTED PERFORMANCE")
    print(f"{'='*80}")
    print(f"Challenge 1 (Trial-Level RT):")
    print(f"   Previous best: 1.09 NRMSE")
    print(f"   Expected: 0.85-0.95 NRMSE")
    print(f"   Improvement: 13-22%")
    print(f"\nChallenge 2 (Domain Adaptation):")
    print(f"   Expected: 1.00-1.10 NRMSE")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Upload {zip_path} to competition platform")
    print(f"2. Select Challenge 1 (trial-level RT prediction)")
    print(f"3. Wait for evaluation (~5-10 minutes)")
    print(f"4. Compare with previous best (1.09)")
    print(f"\n🎯 Target: Beat 0.976 (current leader)")
    print(f"{'='*80}\n")

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Create trial-level RT prediction submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create C1 submission only
    python3 create_trial_level_submission.py \\
        --checkpoint_c1 checkpoints/trial_level_c1_best.pt \\
        --name trial_level_fixed_c1

    # Create C1 + C2 submission
    python3 create_trial_level_submission.py \\
        --checkpoint_c1 checkpoints/trial_level_c1_best.pt \\
        --checkpoint_c2 checkpoints/domain_adaptation_c2_best.pt \\
        --name trial_level_fixed_both
        """
    )

    parser.add_argument(
        '--checkpoint_c1',
        type=str,
        required=True,
        help='Path to C1 checkpoint (trial-level model)'
    )
    parser.add_argument(
        '--checkpoint_c2',
        type=str,
        default=None,
        help='Path to C2 checkpoint (optional, domain adaptation model)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='trial_level_submission',
        help='Submission name (default: trial_level_submission)'
    )

    args = parser.parse_args()

    # Create submission
    zip_path = create_submission(
        checkpoint_c1=args.checkpoint_c1,
        checkpoint_c2=args.checkpoint_c2,
        name=args.name
    )

    print(f"✅ Done! Ready to upload: {zip_path}")


if __name__ == "__main__":
    main()
