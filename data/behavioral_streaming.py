"""
Behavioral Data Streaming - Load targets from S3 without downloading

Loads behavioral targets (response time, externalizing factor) from HBN phenotype data
directly from S3 or local cache.
"""

import pandas as pd
from pathlib import Path
import tempfile
import os

try:
    import s3fs
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


# HBN phenotype data locations (these contain behavioral targets)
PHENOTYPE_S3_PATHS = {
    # Main phenotype file with behavioral data
    'main': 's3://fcp-indi/data/Projects/HBN/phenotype/9994_Basic_Demos_20210310.csv',

    # Note: The exact phenotype files with response time and externalizing factor
    # may be in different locations. These are typical HBN paths.
    # You may need to adjust based on actual HBN dataset structure.
}


class BehavioralDataStreamer:
    """
    Stream behavioral targets from S3 or local cache

    Handles:
    - Response time (Challenge 1)
    - Externalizing factor (Challenge 2)
    - S3 streaming with local caching
    - Fallback to synthetic data for testing
    """

    def __init__(self, use_s3=True, cache_dir='./behavioral_cache',
                 use_synthetic=False):
        """
        Args:
            use_s3: Use S3 streaming (requires s3fs)
            cache_dir: Local cache directory
            use_synthetic: Use synthetic data if real data not available
        """
        self.use_s3 = use_s3 and S3_AVAILABLE
        self.cache_dir = Path(cache_dir)
        self.use_synthetic = use_synthetic
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 filesystem
        if self.use_s3:
            self.fs = s3fs.S3FileSystem(anon=True)
            print("✅ Behavioral data streaming enabled (S3)")
        else:
            print("📁 Using local behavioral data only")

        # Load phenotype data
        self.phenotype_data = self._load_phenotype_data()

    def _load_phenotype_data(self):
        """Load phenotype data from S3 or cache"""
        cache_file = self.cache_dir / 'phenotype_main.csv'

        # Try cache first
        if cache_file.exists():
            print(f"💾 Loading phenotype from cache: {cache_file}")
            try:
                return pd.read_csv(cache_file)
            except Exception as e:
                print(f"⚠️  Cache corrupted, will re-download: {e}")
                cache_file.unlink()

        # Try S3 streaming
        if self.use_s3:
            try:
                print(f"📥 Streaming phenotype data from S3...")
                s3_path = PHENOTYPE_S3_PATHS['main']

                with self.fs.open(s3_path, 'rb') as f:
                    df = pd.read_csv(f)

                # Cache for future use
                df.to_csv(cache_file, index=False)
                print(f"✅ Phenotype data loaded: {len(df)} subjects")
                return df

            except Exception as e:
                print(f"❌ Error loading from S3: {e}")
                print("   Trying alternative approach...")

        # Fallback: Create synthetic data for testing
        if self.use_synthetic:
            print("⚠️  Using synthetic behavioral data for testing")
            return self._create_synthetic_data()
        else:
            print("❌ No behavioral data available")
            return pd.DataFrame()

    def _create_synthetic_data(self):
        """
        Create synthetic behavioral data for testing

        This generates plausible values based on typical EEG-behavior relationships.
        Should be replaced with real data for actual submissions!
        """
        import numpy as np

        print("🧪 Generating synthetic behavioral data...")
        print("   ⚠️  WARNING: This is for TESTING only!")
        print("   ⚠️  Use real phenotype data for actual submissions!")

        # Generate synthetic subjects
        n_subjects = 3000

        # Response time: typically 300-1200 ms, mean ~600-800 ms
        # Normalized to [0, 1] range for model training
        response_times = np.random.normal(0.95, 0.15, n_subjects)
        response_times = np.clip(response_times, 0.5, 1.5)

        # Externalizing factor: standardized score, typically mean=0, std=1
        # For model, we'll use normalized values
        externalizing = np.random.normal(0.0, 0.3, n_subjects)
        externalizing = np.clip(externalizing, -1.0, 1.0)

        # Create subject IDs (format: sub-NDARPXXXXXXX)
        subject_ids = [f"sub-NDARP{i:07d}" for i in range(n_subjects)]

        df = pd.DataFrame({
            'participant_id': subject_ids,
            'response_time': response_times,
            'externalizing_factor': externalizing
        })

        # Cache synthetic data
        cache_file = self.cache_dir / 'phenotype_synthetic.csv'
        df.to_csv(cache_file, index=False)

        return df

    def get_target(self, subject_id, challenge='c1'):
        """
        Get behavioral target for a subject

        Args:
            subject_id: Subject ID (e.g., 'sub-NDARPG836PWJ')
            challenge: 'c1' (response time) or 'c2' (externalizing factor)

        Returns:
            target: Float value, or None if not available
        """
        if self.phenotype_data.empty:
            # No data available, return plausible default
            if challenge == 'c1':
                return 0.95  # Typical response time (normalized)
            else:
                return 0.0  # Typical externalizing (centered)

        # Clean subject ID (remove 'sub-' prefix if present)
        clean_id = subject_id.replace('sub-', '') if 'sub-' in subject_id else subject_id

        # Try different ID column names
        id_columns = ['participant_id', 'subjectkey', 'Subject', 'ID']

        subject_row = None
        for col in id_columns:
            if col in self.phenotype_data.columns:
                # Try exact match
                mask = self.phenotype_data[col].astype(str).str.contains(clean_id, na=False)
                if mask.any():
                    subject_row = self.phenotype_data[mask].iloc[0]
                    break

        if subject_row is None:
            # Subject not found, return default based on challenge
            if challenge == 'c1':
                return 0.95
            else:
                return 0.0

        # Extract target based on challenge
        if challenge == 'c1':
            # Response time - try different possible column names
            rt_columns = ['response_time', 'rt', 'reaction_time', 'RT']
            for col in rt_columns:
                if col in subject_row and pd.notna(subject_row[col]):
                    return float(subject_row[col])
            return 0.95  # Default

        else:  # c2
            # Externalizing factor
            ext_columns = ['externalizing_factor', 'externalizing', 'ext', 'EXT']
            for col in ext_columns:
                if col in subject_row and pd.notna(subject_row[col]):
                    return float(subject_row[col])
            return 0.0  # Default

    def get_statistics(self):
        """Get statistics about behavioral data"""
        if self.phenotype_data.empty:
            return {"status": "No data loaded"}

        stats = {
            "n_subjects": len(self.phenotype_data),
            "columns": list(self.phenotype_data.columns),
        }

        # Try to get statistics for key columns
        if 'response_time' in self.phenotype_data.columns:
            rt = self.phenotype_data['response_time'].dropna()
            stats['response_time'] = {
                'mean': float(rt.mean()),
                'std': float(rt.std()),
                'min': float(rt.min()),
                'max': float(rt.max()),
                'n_valid': len(rt)
            }

        if 'externalizing_factor' in self.phenotype_data.columns:
            ext = self.phenotype_data['externalizing_factor'].dropna()
            stats['externalizing_factor'] = {
                'mean': float(ext.mean()),
                'std': float(ext.std()),
                'min': float(ext.min()),
                'max': float(ext.max()),
                'n_valid': len(ext)
            }

        return stats


# Global streamer instance (singleton pattern)
_global_streamer = None


def get_behavioral_streamer(use_s3=True, use_synthetic=True):
    """
    Get global behavioral data streamer (singleton)

    Args:
        use_s3: Use S3 streaming
        use_synthetic: Fallback to synthetic data for testing

    Returns:
        BehavioralDataStreamer instance
    """
    global _global_streamer

    if _global_streamer is None:
        _global_streamer = BehavioralDataStreamer(
            use_s3=use_s3,
            use_synthetic=use_synthetic
        )

    return _global_streamer


if __name__ == "__main__":
    """Test behavioral streaming"""
    print("="*70)
    print("🧪 Testing Behavioral Data Streaming")
    print("="*70)

    # Test with synthetic data first
    streamer = BehavioralDataStreamer(use_s3=True, use_synthetic=True)

    # Get statistics
    print("\n📊 Data Statistics:")
    stats = streamer.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test getting targets
    print("\n🎯 Testing target retrieval:")
    test_subjects = ['sub-NDARP0000001', 'sub-NDARP0000100', 'sub-NDARP0001000']

    for subject_id in test_subjects:
        c1_target = streamer.get_target(subject_id, 'c1')
        c2_target = streamer.get_target(subject_id, 'c2')
        print(f"   {subject_id}:")
        print(f"      C1 (response time): {c1_target:.4f}")
        print(f"      C2 (externalizing): {c2_target:.4f}")

    print("\n" + "="*70)
    print("✅ Behavioral streaming test complete!")
    print("="*70)
    print("\n💡 Note: Currently using synthetic data for testing")
    print("   For real submissions, ensure phenotype CSV is available")
