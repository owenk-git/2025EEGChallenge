"""
COMPREHENSIVE C1 TARGET INVESTIGATION

Current problem: Trained model gets C1: 1.36, random weights got 0.93
We need to find what C1 actually predicts!

This script checks:
1. Does EEGChallengeDataset provide .y targets?
2. What's in the description table columns?
3. What values do contrastchangedetection_1/2/3 columns have?
4. What does random initialization actually output?
5. What are the actual RT statistics across all data?
"""

from eegdash.dataset import EEGChallengeDataset
import numpy as np
import torch
import torch.nn as nn

print("="*70)
print("COMPREHENSIVE C1 TARGET INVESTIGATION")
print("="*70)

# Load dataset
print("\n1️⃣ Loading EEGChallengeDataset...")
dataset = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R11",
    cache_dir="./data_cache/eeg_challenge",
    mini=False
)

print(f"   Loaded {len(dataset.datasets)} recordings")
print(f"   Subjects: {dataset.description['subject'].nunique()}")

# Check 1: Does dataset have .y?
print("\n2️⃣ Checking for built-in targets (.y attribute)...")
print(f"   Has .y: {hasattr(dataset, 'y')}")
if hasattr(dataset, 'y') and dataset.y is not None:
    print(f"   ✅ FOUND BUILT-IN TARGETS!")
    print(f"   Shape: {dataset.y.shape}")
    print(f"   Type: {type(dataset.y)}")
    print(f"   Sample (first 20): {dataset.y[:20]}")
    print(f"   Statistics:")
    print(f"     Mean: {np.mean(dataset.y):.4f}")
    print(f"     Std:  {np.std(dataset.y):.4f}")
    print(f"     Min:  {np.min(dataset.y):.4f}")
    print(f"     Max:  {np.max(dataset.y):.4f}")
    print(f"\n   ⚠️  WE SHOULD USE THESE TARGETS INSTEAD OF RT EXTRACTION!")
else:
    print(f"   ❌ No .y attribute found")

# Check 2: Description table columns
print("\n3️⃣ Analyzing description table columns...")
print(f"   Total columns: {len(dataset.description.columns)}")

# Check contrastchangedetection columns
ccd_cols = [col for col in dataset.description.columns if 'contrastchange' in col.lower()]
print(f"\n   Contrast Change Detection columns: {ccd_cols}")

for col in ccd_cols:
    unique_vals = dataset.description[col].unique()
    print(f"\n   Column: {col}")
    print(f"     Unique values: {unique_vals}")
    print(f"     Count: {len(unique_vals)}")

    # If numeric, show stats
    if dataset.description[col].dtype in ['float64', 'int64']:
        print(f"     Type: NUMERIC")
        vals = dataset.description[col].dropna()
        print(f"     Mean: {vals.mean():.4f}")
        print(f"     Std:  {vals.std():.4f}")
        print(f"     Range: [{vals.min():.4f}, {vals.max():.4f}]")
    else:
        print(f"     Type: {dataset.description[col].dtype}")

# Check 3: All behavioral/performance columns
print("\n4️⃣ Checking ALL potentially useful columns...")
interesting_cols = [
    'p_factor', 'attention', 'internalizing', 'externalizing',
    'ehq_total', 'age', 'sex'
]

for col in interesting_cols:
    if col in dataset.description.columns:
        vals = dataset.description[col].dropna()
        if len(vals) > 0:
            print(f"\n   {col}:")
            if vals.dtype in ['float64', 'int64']:
                print(f"     Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")
                print(f"     Range: [{vals.min():.4f}, {vals.max():.4f}]")
            else:
                print(f"     Unique values: {vals.unique()[:10]}")

# Check 4: What does random model output?
print("\n5️⃣ Testing random model outputs...")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_conv = nn.Conv1d(129, 64, kernel_size=25, padding=12)
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

        # Init like Oct 14 submission
        nn.init.xavier_normal_(self.fc1.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc2.weight, gain=0.5)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0.5)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.5)

    def forward(self, x):
        x = torch.relu(self.bn(self.temporal_conv(x)))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        # Oct 14 scaling
        x = 0.5 + x * 1.0
        return x

model = SimpleModel()
model.eval()

# Test on random data (100 samples)
random_outputs = []
for _ in range(100):
    fake_data = torch.randn(1, 129, 200)
    with torch.no_grad():
        out = model(fake_data)
    random_outputs.append(out.item())

print(f"   Random model outputs (100 samples):")
print(f"     Mean: {np.mean(random_outputs):.4f}")
print(f"     Std:  {np.std(random_outputs):.4f}")
print(f"     Range: [{np.min(random_outputs):.4f}, {np.max(random_outputs):.4f}]")
print(f"\n   🤔 Random outputs centered around {np.mean(random_outputs):.2f}")
print(f"      If true C1 targets are also ~{np.mean(random_outputs):.2f}, random would work!")

# Check 5: RT extraction statistics across more data
print("\n6️⃣ RT extraction statistics (first 50 recordings)...")
from data.rt_extractor import extract_response_time

rt_values = []
for i in range(min(50, len(dataset.datasets))):
    raw = dataset.datasets[i].raw
    rt = extract_response_time(raw, method='mean', verbose=False)
    if rt is not None:
        rt_values.append(rt)

if len(rt_values) > 0:
    print(f"   Successful extractions: {len(rt_values)}/50")
    print(f"   RT statistics:")
    print(f"     Mean: {np.mean(rt_values):.4f}s")
    print(f"     Std:  {np.std(rt_values):.4f}s")
    print(f"     Range: [{np.min(rt_values):.4f}, {np.max(rt_values):.4f}]s")

    # Show normalized
    rt_norm = [(rt - 1.0) / 1.0 for rt in rt_values]
    print(f"   Normalized [1.0-2.0] RT:")
    print(f"     Mean: {np.mean(rt_norm):.4f}")
    print(f"     Range: [{np.min(rt_norm):.4f}, {np.max(rt_norm):.4f}]")

print("\n" + "="*70)
print("INVESTIGATION COMPLETE - KEY FINDINGS:")
print("="*70)
print("Check above for:")
print("1. Whether dataset.y exists (CRITICAL!)")
print("2. What contrastchangedetection columns contain")
print("3. What random model outputs (~1.0?)")
print("4. RT extraction statistics")
print("\nNext step: Use findings to fix C1 targets!")
print("="*70)
