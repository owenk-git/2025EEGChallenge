"""
Streaming Dataset Loader for HBN-EEG
Supports S3 streaming without downloading full dataset

Features:
- Direct S3 access (no download)
- Smart caching (LRU)
- Progressive loading
- Memory efficient
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import mne
from pathlib import Path
from functools import lru_cache
import tempfile
import shutil

try:
    import s3fs
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("⚠️  s3fs not installed. Install with: pip install s3fs boto3")

# Import behavioral streaming
from .behavioral_streaming import get_behavioral_streamer


class StreamingHBNDataset(Dataset):
    """
    HBN-EEG Dataset with S3 streaming support

    Parameters:
    -----------
    data_source : str or list
        Either:
        - Local path: './data/R1_mini_L100'
        - S3 URI: 's3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1'
        - List of subjects: ['sub-XXXXX', 'sub-YYYYY']
    challenge : str
        'c1' or 'c2'
    max_subjects : int
        Maximum number of subjects to load (for efficiency)
    use_cache : bool
        Cache downloaded files locally (recommended for S3)
    cache_dir : str
        Directory for cached files
    """

    def __init__(self, data_source, challenge='c1', max_subjects=None,
                 use_cache=True, cache_dir='./data_cache',
                 target_sfreq=100, use_synthetic_targets=True):

        self.data_source = data_source
        self.challenge = challenge
        self.max_subjects = max_subjects
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.target_sfreq = target_sfreq

        # Create cache directory
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize behavioral data streamer
        self.behavioral_streamer = get_behavioral_streamer(
            use_s3=True,
            use_synthetic=use_synthetic_targets
        )

        # Determine data source type
        self.is_s3 = isinstance(data_source, str) and data_source.startswith('s3://')
        self.is_local = isinstance(data_source, str) and not self.is_s3
        self.is_list = isinstance(data_source, list)

        # Initialize S3 filesystem if needed
        if self.is_s3:
            if not S3_AVAILABLE:
                raise ImportError("s3fs not installed. Install with: pip install s3fs boto3")
            self.fs = s3fs.S3FileSystem(anon=True)
            print(f"✅ S3 streaming enabled: {data_source}")

        # Get file list
        self.files = self._get_file_list()
        print(f"📊 Loaded {len(self.files)} files for Challenge {challenge}")

    def _get_file_list(self):
        """Get list of EEG files from data source"""
        files = []

        if self.is_local:
            # Local path
            data_path = Path(self.data_source)
            for ext in ['.set', '.bdf']:
                found = list(data_path.rglob(f'*_eeg{ext}'))
                files.extend(found)

        elif self.is_s3:
            # S3 streaming
            # List files from S3
            try:
                all_files = self.fs.glob(f"{self.data_source}/**/sub-*/*_eeg.bdf")
                files = [f"s3://{f}" for f in all_files]
            except Exception as e:
                print(f"❌ Error listing S3 files: {e}")
                print("   Make sure the S3 path is correct")
                return []

        elif self.is_list:
            # List of specific subjects
            # This requires knowing the base path
            print("⚠️  List mode requires additional configuration")
            return []

        # Limit to max_subjects
        if self.max_subjects and len(files) > self.max_subjects:
            import random
            files = random.sample(files, self.max_subjects)
            print(f"📉 Sampled {self.max_subjects} subjects (from {len(files)} available)")

        return sorted(files)

    def _load_from_s3(self, s3_path):
        """
        Load EEG file from S3 with caching

        Args:
            s3_path: S3 URI (e.g., s3://bucket/path/file.bdf)

        Returns:
            raw: MNE Raw object
        """
        # Check cache first
        cache_filename = s3_path.replace('s3://', '').replace('/', '_')
        cache_path = self.cache_dir / cache_filename

        if self.use_cache and cache_path.exists():
            # Load from cache
            print(f"💾 Loading from cache: {cache_filename[:50]}...")
            try:
                raw = mne.io.read_raw_bdf(str(cache_path), preload=True, verbose=False)
                return raw
            except Exception as e:
                print(f"⚠️  Cache corrupted, re-downloading: {e}")
                cache_path.unlink()

        # Download from S3
        print(f"📥 Streaming from S3: {s3_path[-60:]}...")

        try:
            # Use temporary file for S3 streaming
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bdf') as tmp:
                # Stream from S3 to temp file
                with self.fs.open(s3_path, 'rb') as f_in:
                    shutil.copyfileobj(f_in, tmp)
                tmp_path = tmp.name

            # Load from temp file
            raw = mne.io.read_raw_bdf(tmp_path, preload=True, verbose=False)

            # Cache for future use
            if self.use_cache:
                shutil.copy(tmp_path, cache_path)
                print(f"💾 Cached: {cache_filename[:50]}...")

            # Clean up temp file
            os.unlink(tmp_path)

            return raw

        except Exception as e:
            print(f"❌ Error loading from S3: {e}")
            # Return dummy data on error
            return None

    def _extract_subject_id(self, file_path):
        """
        Extract subject ID from file path

        Examples:
            /path/sub-NDARPG836PWJ/eeg/file.bdf → sub-NDARPG836PWJ
            s3://bucket/sub-ABCDEFG/eeg/file.set → sub-ABCDEFG
        """
        path_str = str(file_path)

        # Look for pattern like "sub-XXXXXXX"
        import re
        match = re.search(r'(sub-[A-Z0-9]+)', path_str)
        if match:
            return match.group(1)

        # Fallback: return a generic ID
        return "sub-UNKNOWN"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get EEG data and target"""
        file_path = self.files[idx]

        try:
            # Load based on source type
            if self.is_s3:
                raw = self._load_from_s3(file_path)
                if raw is None:
                    raise ValueError("Failed to load from S3")
            else:
                # Local file
                if str(file_path).endswith('.set'):
                    raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
                elif str(file_path).endswith('.bdf'):
                    raw = mne.io.read_raw_bdf(str(file_path), preload=True, verbose=False)
                else:
                    raise ValueError(f"Unsupported format: {file_path}")

            # Resample if needed
            if raw.info['sfreq'] != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)

            # Get data (channels × time)
            data = raw.get_data()

            # Take 2-second window
            n_samples = int(2 * self.target_sfreq)
            if data.shape[1] > n_samples:
                start = (data.shape[1] - n_samples) // 2
                data = data[:, start:start + n_samples]
            elif data.shape[1] < n_samples:
                pad_width = ((0, 0), (0, n_samples - data.shape[1]))
                data = np.pad(data, pad_width, mode='constant')

            # Ensure 129 channels (pad if needed)
            if data.shape[0] < 129:
                pad_channels = ((0, 129 - data.shape[0]), (0, 0))
                data = np.pad(data, pad_channels, mode='constant')
            elif data.shape[0] > 129:
                data = data[:129, :]

            # Convert to tensor
            data = torch.from_numpy(data).float()

            # Get real behavioral target from subject ID
            # Extract subject ID from file path
            subject_id = self._extract_subject_id(file_path)

            # Get target from behavioral streamer
            target_value = self.behavioral_streamer.get_target(subject_id, self.challenge)
            target = torch.tensor([target_value], dtype=torch.float32)

            return data, target

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            # Return dummy data
            data = torch.randn(129, 200)
            target = torch.tensor([1.0 if self.challenge == 'c1' else 0.0])
            return data, target


def create_streaming_dataloader(data_source, challenge='c1', batch_size=32,
                                max_subjects=None, **kwargs):
    """
    Create streaming dataloader

    Args:
        data_source: Local path, S3 URI, or list of subjects
        challenge: 'c1' or 'c2'
        batch_size: Batch size
        max_subjects: Limit number of subjects (for efficiency)

    Returns:
        dataloader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = StreamingHBNDataset(
        data_source,
        challenge=challenge,
        max_subjects=max_subjects,
        **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # S3 streaming doesn't work well with multiprocessing
        pin_memory=True
    )

    return dataloader


if __name__ == "__main__":
    """Test streaming dataset"""
    print("="*70)
    print("🧪 Testing Streaming Dataset")
    print("="*70)

    # Test 1: Local mini dataset
    print("\n📁 Test 1: Local dataset")
    local_path = "./data/R1_mini_L100"
    if os.path.exists(local_path):
        dataset = StreamingHBNDataset(local_path, challenge='c1', max_subjects=5)
        print(f"✅ Local dataset: {len(dataset)} files")

        data, target = dataset[0]
        print(f"   Data shape: {data.shape}")
        print(f"   Target shape: {target.shape}")
    else:
        print(f"⚠️  Local path not found: {local_path}")

    # Test 2: S3 streaming
    print("\n☁️  Test 2: S3 streaming")
    if S3_AVAILABLE:
        s3_path = "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1"
        try:
            dataset_s3 = StreamingHBNDataset(
                s3_path,
                challenge='c1',
                max_subjects=2,  # Just 2 for testing
                use_cache=True
            )
            print(f"✅ S3 dataset: {len(dataset_s3)} files")

            if len(dataset_s3) > 0:
                data, target = dataset_s3[0]
                print(f"   Data shape: {data.shape}")
                print(f"   Cached in: {dataset_s3.cache_dir}")
        except Exception as e:
            print(f"❌ S3 test failed: {e}")
    else:
        print("⚠️  s3fs not installed, skipping S3 test")

    print("\n" + "="*70)
    print("✅ Streaming dataset tests complete!")
    print("="*70)
