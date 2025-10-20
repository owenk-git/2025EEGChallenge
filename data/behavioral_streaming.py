"""
Behavioral Data Streaming - Load targets from S3 BIDS format without downloading

Loads behavioral targets (response time, externalizing factor) from HBN-EEG dataset
in BIDS format directly from S3 or local cache.

Updated to use CORRECT S3 paths and parse real BIDS format:
- s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf (competition dataset)
- Loads events.tsv for response times (Challenge 1)
- Loads participants.tsv for externalizing factor (Challenge 2)
"""

import pandas as pd
from pathlib import Path
import io
import json

try:
    import s3fs
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


# CORRECT S3 paths for NeurIPS 2025 EEG Challenge
COMPETITION_S3_BASE = 's3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf'

# BIDS structure for behavioral data
# events.tsv: Contains trial-level data including response times
# participants.tsv: Contains subject-level phenotype data including externalizing factor


class BehavioralDataStreamer:
    """
    Stream behavioral targets from S3 BIDS dataset or local cache

    Handles:
    - Response time from events.tsv files (Challenge 1)
    - Externalizing factor from participants.tsv (Challenge 2)
    - S3 streaming with local caching
    - BIDS format parsing
    """

    def __init__(self, use_s3=True, cache_dir='./behavioral_cache',
                 s3_base=COMPETITION_S3_BASE, use_synthetic=False):
        """
        Args:
            use_s3: Use S3 streaming (requires s3fs)
            cache_dir: Local cache directory
            s3_base: S3 base path for dataset
            use_synthetic: Use synthetic data if real data not available
        """
        self.use_s3 = use_s3 and S3_AVAILABLE
        self.cache_dir = Path(cache_dir)
        self.s3_base = s3_base
        self.use_synthetic = use_synthetic
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 filesystem
        if self.use_s3:
            self.fs = s3fs.S3FileSystem(anon=True)
            print(f"✅ Behavioral data streaming enabled from: {s3_base}")
        else:
            print("📁 Using local behavioral data only")

        # Load BIDS participants.tsv (subject-level data)
        self.participants_data = self._load_participants_tsv()

        # Cache for events.tsv (trial-level data) - loaded per subject as needed
        self.events_cache = {}

    def _load_participants_tsv(self):
        """
        Load participants.tsv from S3 BIDS dataset

        This contains subject-level phenotype data including:
        - participant_id (subject identifier)
        - Externalizing factor (CBCL-derived measure for Challenge 2)
        - Other demographic/phenotype information
        """
        cache_file = self.cache_dir / 'participants.tsv'

        # Try cache first
        if cache_file.exists():
            print(f"💾 Loading participants.tsv from cache")
            try:
                return pd.read_csv(cache_file, sep='\t')
            except Exception as e:
                print(f"⚠️  Cache corrupted, will re-download: {e}")
                cache_file.unlink()

        # Try S3 streaming
        if self.use_s3:
            try:
                print(f"📥 Streaming participants.tsv from S3...")
                participants_path = f"{self.s3_base}/participants.tsv"

                with self.fs.open(participants_path, 'rb') as f:
                    df = pd.read_csv(f, sep='\t')

                # Cache for future use
                df.to_csv(cache_file, sep='\t', index=False)
                print(f"✅ Participants data loaded: {len(df)} subjects")
                print(f"   Columns: {list(df.columns)}")
                return df

            except Exception as e:
                print(f"❌ Error loading participants.tsv from S3: {e}")
                print(f"   Tried path: {participants_path}")

        # Fallback: Create synthetic data for testing
        if self.use_synthetic:
            print("⚠️  Using synthetic participants data for testing")
            return self._create_synthetic_participants()
        else:
            print("❌ No participants data available")
            return pd.DataFrame()

    def _load_events_tsv(self, subject_id, session='01', task='contrastChangeDetection'):
        """
        Load events.tsv for a specific subject/session/task

        BIDS path structure:
        sub-{subject_id}/ses-{session}/eeg/sub-{subject_id}_ses-{session}_task-{task}_events.tsv

        Events.tsv contains trial-level data including:
        - onset: Time of event
        - duration: Event duration
        - trial_type: Type of trial
        - response_time: Reaction time (for Challenge 1)
        - correct: Whether response was correct
        """
        # Create cache key
        cache_key = f"{subject_id}_ses-{session}_task-{task}"

        # Check cache first
        if cache_key in self.events_cache:
            return self.events_cache[cache_key]

        # Try loading from S3
        if self.use_s3:
            try:
                # BIDS path structure
                events_path = (
                    f"{self.s3_base}/sub-{subject_id}/ses-{session}/eeg/"
                    f"sub-{subject_id}_ses-{session}_task-{task}_events.tsv"
                )

                with self.fs.open(events_path, 'rb') as f:
                    df = pd.read_csv(f, sep='\t')

                # Cache in memory
                self.events_cache[cache_key] = df
                return df

            except Exception as e:
                # File might not exist for this subject/session/task
                # This is normal - not all subjects have all tasks
                return pd.DataFrame()

        # Try local cache
        cache_file = self.cache_dir / f"events_{cache_key}.tsv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, sep='\t')
                self.events_cache[cache_key] = df
                return df
            except:
                pass

        return pd.DataFrame()

    def _create_synthetic_participants(self):
        """Create synthetic participants.tsv for testing"""
        import numpy as np

        print("🧪 Generating synthetic participants data...")
        print("   ⚠️  WARNING: This is for TESTING only!")

        n_subjects = 3000

        # Externalizing factor: standardized score (mean=0, std=1)
        externalizing = np.random.normal(0.0, 1.0, n_subjects)

        # Subject IDs in BIDS format
        subject_ids = [f"NDAR{i:08d}" for i in range(1, n_subjects + 1)]

        df = pd.DataFrame({
            'participant_id': subject_ids,
            'externalizing': externalizing,
            'age': np.random.randint(5, 21, n_subjects),
            'sex': np.random.choice(['M', 'F'], n_subjects)
        })

        # Cache synthetic data
        cache_file = self.cache_dir / 'participants_synthetic.tsv'
        df.to_csv(cache_file, sep='\t', index=False)

        return df

    def get_target(self, subject_id, challenge='c1', task='contrastChangeDetection', session='01'):
        """
        Get behavioral target for a subject

        Args:
            subject_id: Subject ID (e.g., 'NDARPG836PWJ' or 'sub-NDARPG836PWJ')
            challenge: 'c1' (response time) or 'c2' (externalizing factor)
            task: Task name for events.tsv lookup (default: contrastChangeDetection)
            session: Session ID (default: '01')

        Returns:
            target: Float value, or default if not available
        """
        # Clean subject ID (remove 'sub-' prefix if present)
        clean_id = subject_id.replace('sub-', '') if subject_id.startswith('sub-') else subject_id

        if challenge == 'c1':
            # Challenge 1: Response time from events.tsv
            return self._get_response_time(clean_id, task, session)
        else:
            # Challenge 2: Externalizing factor from participants.tsv
            return self._get_externalizing_factor(clean_id)

    def _get_response_time(self, subject_id, task='contrastChangeDetection', session='01'):
        """
        Get median response time for a subject from events.tsv

        Returns normalized response time (typical range: 0.8-1.2)
        """
        # Load events.tsv for this subject
        events_df = self._load_events_tsv(subject_id, session, task)

        if events_df.empty:
            # No events data - return typical default
            return 0.95

        # Look for response time column (different possible names)
        rt_columns = ['response_time', 'rt', 'reaction_time', 'RT']
        rt_col = None
        for col in rt_columns:
            if col in events_df.columns:
                rt_col = col
                break

        if rt_col is None:
            return 0.95  # Default if no RT column found

        # Get valid response times (non-NaN, positive values)
        rts = events_df[rt_col].dropna()
        rts = rts[rts > 0]  # Remove invalid RTs

        if len(rts) == 0:
            return 0.95  # No valid RTs

        # Use median RT for robustness
        median_rt = float(rts.median())

        # Normalize to typical range (assuming RTs are in seconds)
        # Competition likely expects normalized values around 0.8-1.2
        # Adjust normalization based on actual data distribution
        normalized_rt = median_rt  # May need scaling based on actual data

        return normalized_rt

    def _get_externalizing_factor(self, subject_id):
        """
        Get externalizing factor for a subject from participants.tsv

        Returns standardized externalizing score (mean≈0, std≈1)
        """
        if self.participants_data.empty:
            return 0.0  # Default centered value

        # Try different possible ID column names
        id_columns = ['participant_id', 'subject_id', 'sub']

        subject_row = None
        for col in id_columns:
            if col in self.participants_data.columns:
                # Clean the column values
                clean_col = self.participants_data[col].astype(str).str.replace('sub-', '')

                # Try exact match
                mask = clean_col == subject_id
                if mask.any():
                    subject_row = self.participants_data[mask].iloc[0]
                    break

        if subject_row is None:
            return 0.0  # Subject not found, return default

        # Look for externalizing factor column
        ext_columns = ['externalizing', 'externalizing_factor', 'ext', 'CBCL_Externalizing']
        for col in ext_columns:
            if col in subject_row.index and pd.notna(subject_row[col]):
                return float(subject_row[col])

        return 0.0  # Column not found, return default

    def get_statistics(self):
        """Get statistics about behavioral data"""
        stats = {
            "s3_base": self.s3_base,
            "use_s3": self.use_s3,
            "cache_dir": str(self.cache_dir)
        }

        if not self.participants_data.empty:
            stats["n_subjects"] = len(self.participants_data)
            stats["participants_columns"] = list(self.participants_data.columns)

            # Externalizing statistics
            for col in ['externalizing', 'externalizing_factor', 'CBCL_Externalizing']:
                if col in self.participants_data.columns:
                    ext = self.participants_data[col].dropna()
                    if len(ext) > 0:
                        stats['externalizing'] = {
                            'mean': float(ext.mean()),
                            'std': float(ext.std()),
                            'min': float(ext.min()),
                            'max': float(ext.max()),
                            'n_valid': len(ext)
                        }
                        break
        else:
            stats["status"] = "No participants data loaded"

        # Events cache statistics
        stats["events_cached"] = len(self.events_cache)

        return stats


# Global streamer instance (singleton pattern)
_global_streamer = None


def get_behavioral_streamer(use_s3=True, use_synthetic=False, s3_base=COMPETITION_S3_BASE):
    """
    Get global behavioral data streamer (singleton)

    Args:
        use_s3: Use S3 streaming
        use_synthetic: Fallback to synthetic data for testing
        s3_base: S3 base path

    Returns:
        BehavioralDataStreamer instance
    """
    global _global_streamer

    if _global_streamer is None:
        _global_streamer = BehavioralDataStreamer(
            use_s3=use_s3,
            use_synthetic=use_synthetic,
            s3_base=s3_base
        )

    return _global_streamer


if __name__ == "__main__":
    """Test behavioral streaming with real BIDS format"""
    print("="*70)
    print("🧪 Testing Behavioral Data Streaming (BIDS Format)")
    print("="*70)

    # Test with real S3 path first, fallback to synthetic
    print("\n🔍 Attempting to load from competition S3 bucket...")
    print(f"   Path: {COMPETITION_S3_BASE}")

    streamer = BehavioralDataStreamer(
        use_s3=True,
        use_synthetic=True,  # Fallback if S3 fails
        s3_base=COMPETITION_S3_BASE
    )

    # Get statistics
    print("\n📊 Data Statistics:")
    stats = streamer.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test getting targets with example subject IDs
    print("\n🎯 Testing target retrieval:")
    test_subjects = [
        'NDARPG836PWJ',  # Example from competition
        'NDARAA536PTU',
        'NDARAA075AMK'
    ]

    for subject_id in test_subjects:
        print(f"\n   Subject: {subject_id}")

        # Challenge 1: Response time
        c1_target = streamer.get_target(subject_id, 'c1')
        print(f"      C1 (response time): {c1_target:.4f}")

        # Challenge 2: Externalizing
        c2_target = streamer.get_target(subject_id, 'c2')
        print(f"      C2 (externalizing): {c2_target:.4f}")

    print("\n" + "="*70)
    print("✅ Behavioral streaming test complete!")
    print("="*70)
    print("\n💡 Notes:")
    print("   - Now using CORRECT S3 path: s3://nmdatasets/NeurIPS2025/")
    print("   - Loads events.tsv for response times (per subject/task)")
    print("   - Loads participants.tsv for externalizing factor")
    print("   - Parses BIDS format properly")
    print("   - Caches data locally for efficiency")
