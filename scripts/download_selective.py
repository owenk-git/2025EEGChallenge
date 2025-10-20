"""
Selective Data Downloader - Download only what you need

Strategies:
1. Download N random subjects
2. Download subjects from phenotype criteria
3. Download specific subjects by ID
"""

import argparse
import subprocess
from pathlib import Path
import random

try:
    import s3fs
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


S3_BASE = "s3://fcp-indi/data/Projects/HBN/BIDS_EEG"

RELEASE_PATHS = {
    1: f"{S3_BASE}/cmi_bids_R1",
    2: f"{S3_BASE}/cmi_bids_R2",
    3: f"{S3_BASE}/cmi_bids_R3",
    4: f"{S3_BASE}/cmi_bids_R4",
    5: f"{S3_BASE}/cmi_bids_R5",
    6: f"{S3_BASE}/cmi_bids_R6",
    7: f"{S3_BASE}/cmi_bids_R7",
    8: f"{S3_BASE}/cmi_bids_R8",
    9: f"{S3_BASE}/cmi_bids_R9",
    10: f"{S3_BASE}/cmi_bids_R10",
    11: f"{S3_BASE}/cmi_bids_R11",
}


def list_subjects_in_release(release, max_list=None):
    """List available subjects in a release"""
    if not S3_AVAILABLE:
        print("❌ s3fs not installed. Install with: pip install s3fs boto3")
        return []

    print(f"📋 Listing subjects in Release {release}...")

    fs = s3fs.S3FileSystem(anon=True)
    s3_path = RELEASE_PATHS[release]

    try:
        # List all subject directories
        subjects = fs.glob(f"{s3_path}/sub-*")
        subject_ids = [Path(s).name for s in subjects]

        if max_list:
            subject_ids = subject_ids[:max_list]

        print(f"✅ Found {len(subject_ids)} subjects")
        return subject_ids

    except Exception as e:
        print(f"❌ Error listing subjects: {e}")
        return []


def download_subject(release, subject_id, output_dir='./data'):
    """
    Download a single subject from S3

    Args:
        release: Release number (1-11)
        subject_id: Subject ID (e.g., 'sub-NDARPG836PWJ')
        output_dir: Output directory
    """
    s3_path = f"{RELEASE_PATHS[release]}/{subject_id}"
    local_path = Path(output_dir) / f"R{release}" / subject_id

    print(f"📥 Downloading {subject_id} from Release {release}...")

    # Create output directory
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Use AWS CLI to download
    cmd = [
        "aws", "s3", "cp",
        s3_path,
        str(local_path),
        "--recursive",
        "--no-sign-request"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Downloaded to: {local_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading {subject_id}: {e}")
        return False
    except FileNotFoundError:
        print("❌ AWS CLI not found. Install with: pip install awscli")
        return False


def download_random_subjects(release, n_subjects, output_dir='./data'):
    """Download N random subjects from a release"""
    print("="*70)
    print(f"📥 Downloading {n_subjects} random subjects from Release {release}")
    print("="*70)

    # List all subjects
    subjects = list_subjects_in_release(release)

    if not subjects:
        print("❌ No subjects found")
        return

    # Sample randomly
    if len(subjects) > n_subjects:
        selected = random.sample(subjects, n_subjects)
    else:
        selected = subjects
        print(f"⚠️  Only {len(subjects)} subjects available")

    print(f"\n📋 Selected subjects:")
    for s in selected:
        print(f"   - {s}")

    print(f"\n📥 Starting downloads...")

    success = 0
    for i, subject_id in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {subject_id}")
        if download_subject(release, subject_id, output_dir):
            success += 1

    print("\n" + "="*70)
    print(f"✅ Downloaded {success}/{len(selected)} subjects")
    print(f"📁 Location: {output_dir}/R{release}/")
    print("="*70)


def download_specific_subjects(release, subject_ids, output_dir='./data'):
    """Download specific subjects by ID"""
    print("="*70)
    print(f"📥 Downloading {len(subject_ids)} specific subjects")
    print("="*70)

    success = 0
    for i, subject_id in enumerate(subject_ids, 1):
        print(f"\n[{i}/{len(subject_ids)}] {subject_id}")
        if download_subject(release, subject_id, output_dir):
            success += 1

    print("\n" + "="*70)
    print(f"✅ Downloaded {success}/{len(subject_ids)} subjects")
    print("="*70)


def estimate_download_size(n_subjects):
    """Estimate download size"""
    avg_size_mb = 300  # Approximate per subject
    total_mb = n_subjects * avg_size_mb
    total_gb = total_mb / 1024

    print(f"\n💾 Estimated download size:")
    print(f"   {n_subjects} subjects × ~300 MB = ~{total_gb:.1f} GB")

    return total_gb


def main(args):
    """Main function"""

    if args.mode == 'list':
        # Just list subjects
        subjects = list_subjects_in_release(args.release, args.max_list)
        if subjects:
            print(f"\n📋 Subjects in Release {args.release}:")
            for s in subjects[:20]:  # Show first 20
                print(f"   {s}")
            if len(subjects) > 20:
                print(f"   ... and {len(subjects) - 20} more")

    elif args.mode == 'random':
        # Download random subjects
        estimate_download_size(args.n_subjects)

        response = input(f"\nDownload {args.n_subjects} subjects? (y/n): ")
        if response.lower() == 'y':
            download_random_subjects(args.release, args.n_subjects, args.output_dir)
        else:
            print("❌ Cancelled")

    elif args.mode == 'specific':
        # Download specific subjects
        subject_ids = args.subjects
        estimate_download_size(len(subject_ids))

        response = input(f"\nDownload {len(subject_ids)} subjects? (y/n): ")
        if response.lower() == 'y':
            download_specific_subjects(args.release, subject_ids, args.output_dir)
        else:
            print("❌ Cancelled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective data downloader")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['list', 'random', 'specific'],
                        help='Download mode')
    parser.add_argument('--release', type=int, required=True, choices=range(1, 12),
                        help='Release number (1-11)')
    parser.add_argument('--n_subjects', type=int, default=10,
                        help='Number of subjects to download (for random mode)')
    parser.add_argument('--subjects', type=str, nargs='+',
                        help='Specific subject IDs (for specific mode)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--max_list', type=int, default=100,
                        help='Max subjects to list (for list mode)')

    args = parser.parse_args()

    if args.mode == 'specific' and not args.subjects:
        parser.error("--subjects required for specific mode")

    main(args)
