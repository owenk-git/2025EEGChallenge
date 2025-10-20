"""
Quick Test Script for S3 Streaming Training

Tests that S3 streaming + behavioral targets work before full training
"""

import sys
import torch
sys.path.insert(0, '.')

from data.streaming_dataset import create_streaming_dataloader
from models.eegnet import create_model


def test_s3_streaming():
    """Test S3 streaming with behavioral targets"""
    print("="*70)
    print("🧪 Testing S3 Streaming Training Pipeline")
    print("="*70)

    # Configuration
    S3_PATH = "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1"
    MAX_SUBJECTS = 3  # Just 3 for quick test
    BATCH_SIZE = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n📊 Configuration:")
    print(f"   S3 Path: {S3_PATH}")
    print(f"   Max Subjects: {MAX_SUBJECTS}")
    print(f"   Device: {DEVICE}")
    print(f"   Using SYNTHETIC behavioral data for testing")

    # Test Challenge 1
    print("\n" + "="*70)
    print("📊 Testing Challenge 1")
    print("="*70)

    try:
        print("\n1️⃣  Creating dataloader...")
        dataloader = create_streaming_dataloader(
            S3_PATH,
            challenge='c1',
            batch_size=BATCH_SIZE,
            max_subjects=MAX_SUBJECTS,
            use_cache=True,
            cache_dir='./test_cache'
        )
        print(f"✅ Dataloader created: {len(dataloader)} batches")

        print("\n2️⃣  Loading first batch...")
        data, target = next(iter(dataloader))
        print(f"✅ Batch loaded:")
        print(f"   Data shape: {data.shape}")
        print(f"   Target shape: {target.shape}")
        print(f"   Target values: {target.squeeze().tolist()}")
        print(f"   Target range: [{target.min():.3f}, {target.max():.3f}]")

        print("\n3️⃣  Creating model...")
        model = create_model(challenge='c1', device=DEVICE)
        print(f"✅ Model created")

        print("\n4️⃣  Forward pass...")
        output = model(data.to(DEVICE))
        print(f"✅ Forward pass successful:")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

        print("\n5️⃣  Computing loss...")
        criterion = torch.nn.MSELoss()
        loss = criterion(output, target.to(DEVICE))
        print(f"✅ Loss computed: {loss.item():.4f}")

        print("\n6️⃣  Backward pass...")
        loss.backward()
        print(f"✅ Backward pass successful")

        print("\n" + "="*70)
        print("✅ Challenge 1 test PASSED!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Challenge 1 test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Challenge 2
    print("\n" + "="*70)
    print("📊 Testing Challenge 2")
    print("="*70)

    try:
        print("\n1️⃣  Creating dataloader...")
        dataloader = create_streaming_dataloader(
            S3_PATH,
            challenge='c2',
            batch_size=BATCH_SIZE,
            max_subjects=MAX_SUBJECTS,
            use_cache=True,
            cache_dir='./test_cache'
        )
        print(f"✅ Dataloader created")

        print("\n2️⃣  Loading first batch...")
        data, target = next(iter(dataloader))
        print(f"✅ Batch loaded:")
        print(f"   Target values: {target.squeeze().tolist()}")

        print("\n3️⃣  Model forward pass...")
        model = create_model(challenge='c2', device=DEVICE)
        output = model(data.to(DEVICE))
        loss = criterion(output, target.to(DEVICE))
        print(f"✅ Loss: {loss.item():.4f}")

        print("\n" + "="*70)
        print("✅ Challenge 2 test PASSED!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Challenge 2 test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
    print("\n✅ S3 streaming works")
    print("✅ Behavioral targets loaded")
    print("✅ Model forward/backward pass works")
    print("✅ Ready for full training!")
    print("\n📝 Next step:")
    print("   python train.py --challenge 1 \\")
    print("     --data_path s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1 \\")
    print("     --use_streaming --max_subjects 50 --epochs 50")
    print("="*70)

    return True


if __name__ == "__main__":
    success = test_s3_streaming()
    sys.exit(0 if success else 1)
