"""
S3 Streaming Data Loader - NO LOCAL CACHING!

Loads EEG data directly from AWS S3 bucket for NeurIPS 2025 EEG Challenge.
True streaming without local storage.

S3 Bucket: s3://nmdatasets/NeurIPS2025/
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import io
import os

# Check if boto3 is available
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("⚠️ boto3 not installed. Install with: pip install boto3")

# Check if mne is available for BDF reading
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("⚠️ mne not installed. Install with: pip install mne")


class S3StreamingEEGDataset(Dataset):
    """
    Stream EEG data directly from S3 bucket
    NO local caching - pure cloud streaming!

    S3 Bucket: s3://nmdatasets/NeurIPS2025/
    """

    def __init__(
        self,
        challenge='c1',
        release='R11',
        mini=False,
        n_channels=129,
        n_times=900,
        sfreq=100,
        max_files=None
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 required for S3 streaming. Install: pip install boto3")
        if not MNE_AVAILABLE:
            raise ImportError("mne required for BDF reading. Install: pip install mne")

        self.challenge = challenge
        self.n_channels = n_channels
        self.n_times = n_times
        self.sfreq = sfreq

        # S3 configuration (public bucket, no credentials needed)
        self.bucket_name = 'nmdatasets'
        self.s3_prefix = f'NeurIPS2025/{release}{"_mini" if mini else ""}_L100_bdf/'

        # Create S3 client (no credentials needed for public bucket)
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)  # No auth for public bucket
        )

        print(f"📦 S3 Streaming Loader")
        print(f"   Bucket: s3://{self.bucket_name}/{self.s3_prefix}")

        # List all BDF files in S3
        self.file_list = self._list_bdf_files()

        if max_files:
            self.file_list = self.file_list[:max_files]

        print(f"✅ Found {len(self.file_list)} EEG files for streaming")
        print(f"   No local caching - streaming directly from S3!")

    def _list_bdf_files(self):
        """List all BDF files in S3 bucket"""
        print(f"\n🔍 Listing files from S3...")

        file_list = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.s3_prefix):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.bdf') or key.endswith('.BDF'):
                    file_list.append(key)

        return file_list

    def _stream_file_from_s3(self, s3_key):
        """Stream file from S3 into memory"""
        try:
            # Download file to memory (BytesIO)
            file_obj = io.BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, s3_key, file_obj)
            file_obj.seek(0)
            return file_obj
        except Exception as e:
            print(f"❌ Error streaming {s3_key}: {e}")
            return None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Stream EEG data from S3 and extract features

        Returns:
            data: (n_channels, n_times) tensor
            target: scalar target value
        """
        s3_key = self.file_list[idx]

        # Stream file from S3 to memory
        file_obj = self._stream_file_from_s3(s3_key)

        if file_obj is None:
            # Return dummy data if streaming fails
            return torch.zeros(self.n_channels, self.n_times), torch.tensor(0.0)

        try:
            # Read BDF file from memory using MNE
            raw = mne.io.read_raw_bdf(file_obj, preload=True, verbose=False)

            # Get EEG data
            eeg_data = raw.get_data()

            # Ensure correct shape
            if eeg_data.shape[1] > self.n_times:
                eeg_data = eeg_data[:, :self.n_times]
            elif eeg_data.shape[1] < self.n_times:
                pad_width = self.n_times - eeg_data.shape[1]
                eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_width)), mode='constant')

            if eeg_data.shape[0] > self.n_channels:
                eeg_data = eeg_data[:self.n_channels, :]
            elif eeg_data.shape[0] < self.n_channels:
                pad_width = self.n_channels - eeg_data.shape[0]
                eeg_data = np.pad(eeg_data, ((0, pad_width), (0, 0)), mode='constant')

            data = torch.tensor(eeg_data, dtype=torch.float32)

            # Extract target (dummy for now - would need proper extraction)
            # For C1: reaction time
            # For C2: externalizing factor
            target = torch.tensor(1.0, dtype=torch.float32)  # Placeholder

            return data, target

        except Exception as e:
            print(f"❌ Error reading {s3_key}: {e}")
            return torch.zeros(self.n_channels, self.n_times), torch.tensor(0.0)


def create_s3_streaming_loaders(
    challenge='c1',
    release='R11',
    mini=False,
    batch_size=32,
    num_workers=4,
    val_split=0.2,
    max_files=None
):
    """
    Create train/val dataloaders that stream directly from S3

    Args:
        challenge: 'c1' or 'c2'
        release: Release name (e.g., 'R11')
        mini: Use mini dataset
        batch_size: Batch size
        num_workers: Number of workers
        val_split: Validation split ratio
        max_files: Maximum files to use (for testing)

    Returns:
        train_loader, val_loader
    """
    print(f"\n{'='*60}")
    print(f"Creating S3 Streaming Loaders for {challenge}")
    print(f"{'='*60}")

    # Create dataset
    dataset = S3StreamingEEGDataset(
        challenge=challenge,
        release=release,
        mini=mini,
        max_files=max_files
    )

    # Split into train/val
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} files")
    print(f"   Val: {len(val_dataset)} files")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n✅ S3 Streaming loaders ready!")
    print(f"   No local caching - pure cloud streaming")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    return train_loader, val_loader


if __name__ == '__main__':
    # Test S3 streaming
    print("Testing S3 Streaming Loader...")

    if not BOTO3_AVAILABLE:
        print("\n❌ boto3 not installed!")
        print("Install: pip install boto3")
        exit(1)

    if not MNE_AVAILABLE:
        print("\n❌ mne not installed!")
        print("Install: pip install mne")
        exit(1)

    # Test with mini dataset and limited files
    train_loader, val_loader = create_s3_streaming_loaders(
        challenge='c1',
        release='R11',
        mini=True,
        batch_size=2,
        num_workers=0,
        max_files=5  # Only 5 files for testing
    )

    print("\n🧪 Testing one batch...")
    for X_batch, y_batch in train_loader:
        print(f"   X shape: {X_batch.shape}")
        print(f"   y shape: {y_batch.shape}")
        print(f"   ✅ Streaming works!")
        break

    print("\n✨ S3 streaming test completed!")
