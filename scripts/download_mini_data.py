"""
Download mini datasets for HBN-EEG Challenge

Usage:
    python scripts/download_mini_data.py --release 1
    python scripts/download_mini_data.py --releases 1 2 3
"""

import argparse
import gdown
from pathlib import Path


# Google Drive links for mini datasets (100 Hz, 20 subjects)
MINI_DATASETS = {
    1: "https://drive.google.com/drive/folders/R1_mini_L100",  # Placeholder
    2: "https://drive.google.com/drive/folders/R2_mini_L100",
    3: "https://drive.google.com/drive/folders/R3_mini_L100",
    # Add more as needed
}


def download_mini_dataset(release, output_dir='./data'):
    """
    Download mini dataset for a specific release

    Args:
        release: Release number (1-11)
        output_dir: Output directory
    """
    print(f"📥 Downloading R{release}_mini_L100...")

    output_path = Path(output_dir) / f"R{release}_mini_L100"
    output_path.mkdir(parents=True, exist_ok=True)

    # Note: Replace with actual Google Drive links
    if release in MINI_DATASETS:
        print(f"⚠️  Please download manually from:")
        print(f"   https://nemar.org (search for HBN-EEG R{release}_mini_L100)")
        print(f"   Extract to: {output_path}")
    else:
        print(f"❌ Release {release} not available")

    return output_path


def download_from_s3(release, output_dir='./data'):
    """
    Download dataset from S3 (requires AWS CLI)

    Args:
        release: Release number (1-11)
        output_dir: Output directory
    """
    import subprocess

    s3_uri = f"s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R{release}"
    output_path = Path(output_dir) / f"R{release}"

    print(f"📥 Downloading from S3: {s3_uri}")
    print(f"   This may take a while (full release is 100-250 GB)...")

    cmd = [
        "aws", "s3", "cp",
        s3_uri,
        str(output_path),
        "--recursive",
        "--no-sign-request"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Downloaded to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading from S3: {e}")
        print("   Make sure AWS CLI is installed: pip install awscli")


def main(args):
    """Main download function"""
    print("="*70)
    print("📥 HBN-EEG Mini Dataset Downloader")
    print("="*70)

    if args.releases:
        for release in args.releases:
            download_mini_dataset(release, args.output_dir)
            print()
    elif args.release:
        download_mini_dataset(args.release, args.output_dir)

    if args.from_s3:
        print("\n⚠️  WARNING: Full datasets are very large (100-250 GB each)")
        response = input("Continue with S3 download? (y/n): ")
        if response.lower() == 'y':
            for release in args.releases or [args.release]:
                download_from_s3(release, args.output_dir)

    print("\n" + "="*70)
    print("📚 Dataset Download Guide:")
    print("="*70)
    print("Mini datasets (100 Hz, 20 subjects):")
    print("  • Download from: https://nemar.org")
    print("  • Search for: HBN-EEG R#_mini_L100")
    print("  • Extract to: ./data/R#_mini_L100")
    print()
    print("Full datasets (from S3):")
    print("  • Requires AWS CLI: pip install awscli")
    print("  • Run with: --from_s3 flag")
    print("  • Size: 100-250 GB per release")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HBN-EEG datasets")

    parser.add_argument('--release', type=int, choices=range(1, 12),
                        help='Single release to download (1-11)')
    parser.add_argument('--releases', type=int, nargs='+', choices=range(1, 12),
                        help='Multiple releases to download')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--from_s3', action='store_true',
                        help='Download full dataset from S3 (requires AWS CLI)')

    args = parser.parse_args()

    if not args.release and not args.releases:
        parser.error("Provide --release or --releases")

    main(args)
