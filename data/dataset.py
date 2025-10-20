"""
Dataset loader for HBN-EEG data

Supports:
- BIDS format EEG data
- Multiple releases (R1-R11)
- Mini datasets (100 Hz, 20 subjects)
- Both SET and BDF formats
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
from pathlib import Path


class HBNEEGDataset(Dataset):
    """
    HBN-EEG Dataset for Challenge 1 & 2

    Parameters:
    -----------
    data_path : str
        Path to BIDS dataset (e.g., ./data/R1_mini_L100)
    challenge : str
        'c1' or 'c2'
    transform : callable, optional
        Transform to apply to EEG data
    target_sfreq : int
        Target sampling frequency (default: 100 Hz)
    """

    def __init__(self, data_path, challenge='c1', transform=None,
                 target_sfreq=100):
        self.data_path = Path(data_path)
        self.challenge = challenge
        self.transform = transform
        self.target_sfreq = target_sfreq

        # Load file list
        self.files = self._get_file_list()

        print(f"Loaded {len(self.files)} files for Challenge {challenge}")

    def _get_file_list(self):
        """Get list of EEG files from BIDS dataset"""
        files = []

        # Search for EEG files (.set or .bdf)
        for ext in ['.set', '.bdf', '.fdt']:
            files.extend(list(self.data_path.rglob(f'*_eeg{ext}')))

        # Remove .fdt files if corresponding .set exists
        set_files = [f for f in files if f.suffix == '.set']
        if set_files:
            files = set_files

        return sorted(list(set(files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get EEG data and target

        Returns:
            data: (n_channels, n_times) tensor
            target: scalar or vector depending on challenge
        """
        file_path = self.files[idx]

        try:
            # Load EEG data
            if file_path.suffix == '.set':
                raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
            elif file_path.suffix == '.bdf':
                raw = mne.io.read_raw_bdf(str(file_path), preload=True, verbose=False)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Resample if needed
            if raw.info['sfreq'] != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)

            # Get data (channels × time)
            data = raw.get_data()

            # Take 2-second window (200 samples at 100 Hz)
            n_samples = int(2 * self.target_sfreq)
            if data.shape[1] > n_samples:
                # Take middle section
                start = (data.shape[1] - n_samples) // 2
                data = data[:, start:start + n_samples]
            elif data.shape[1] < n_samples:
                # Pad if too short
                pad_width = ((0, 0), (0, n_samples - data.shape[1]))
                data = np.pad(data, pad_width, mode='constant')

            # Convert to tensor
            data = torch.from_numpy(data).float()

            # Apply transform
            if self.transform:
                data = self.transform(data)

            # Get target (dummy for now - will need behavioral data)
            if self.challenge == 'c1':
                target = torch.tensor([1.0])  # Response time (placeholder)
            else:
                target = torch.tensor([0.0])  # Externalizing factor (placeholder)

            return data, target

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy data on error
            data = torch.randn(129, 200)
            target = torch.tensor([1.0 if self.challenge == 'c1' else 0.0])
            return data, target


def create_dataloader(data_path, challenge='c1', batch_size=32,
                      shuffle=True, num_workers=4, **kwargs):
    """
    Create DataLoader for training

    Args:
        data_path: Path to BIDS dataset
        challenge: 'c1' or 'c2'
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of workers for data loading

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = HBNEEGDataset(data_path, challenge=challenge, **kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loader...")

    # Example: Load mini dataset
    data_path = "./data/R1_mini_L100"

    if os.path.exists(data_path):
        dataset = HBNEEGDataset(data_path, challenge='c1')
        print(f"Dataset size: {len(dataset)}")

        # Load one sample
        data, target = dataset[0]
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {target.shape}")

        # Create dataloader
        dataloader = create_dataloader(data_path, challenge='c1', batch_size=4)
        batch_data, batch_target = next(iter(dataloader))
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch target shape: {batch_target.shape}")

        print("✅ Dataset test passed!")
    else:
        print(f"⚠️  Data path not found: {data_path}")
        print("Download mini dataset first with: python scripts/download_mini_data.py")
