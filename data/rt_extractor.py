"""
Response Time (RT) Extraction from EEG Task Events

For Challenge 1, we need to extract actual response times from the
contrast change detection task, not use age as a proxy.

The response time is the interval between:
- Stimulus presentation (e.g., 'stimulus_change' or 'target')
- Subject response (e.g., 'response' or button press)
"""

import numpy as np
import mne


def extract_response_time(raw, method='mean', verbose=False):
    """
    Extract response time from MNE Raw object

    Args:
        raw: MNE Raw object with annotations
        method: 'mean', 'median', 'first' - how to aggregate multiple RTs
        verbose: Print debug info

    Returns:
        Response time in seconds (float), or None if can't extract
    """
    # Get annotations
    annotations = raw.annotations

    if len(annotations) == 0:
        return None

    # Common event descriptions for contrast change detection task
    stimulus_events = [
        'stimulus', 'stimulus_change', 'target', 'stimulus_onset',
        'change', 'detection', 'stimulus_anchor'
    ]

    response_events = [
        'response', 'button', 'keypress', 'reaction',
        'response_left', 'response_right', 'subject_response'
    ]

    # Extract onset times
    event_times = []
    event_descriptions = []

    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        event_times.append(onset)
        event_descriptions.append(description.lower())

    if verbose:
        print(f"Total events: {len(event_descriptions)}")
        print(f"Unique descriptions: {set(event_descriptions)}")

    # Find stimulus-response pairs
    rts = []

    for i, desc in enumerate(event_descriptions):
        # Check if this is a stimulus event
        is_stimulus = any(stim in desc for stim in stimulus_events)

        if is_stimulus:
            # Look for next response event
            for j in range(i + 1, len(event_descriptions)):
                next_desc = event_descriptions[j]
                is_response = any(resp in next_desc for resp in response_events)

                if is_response:
                    rt = event_times[j] - event_times[i]
                    # Sanity check: RT should be between 0.1s and 3s
                    if 0.1 <= rt <= 3.0:
                        rts.append(rt)
                    break

                # If we hit another stimulus before a response, skip this one
                is_next_stimulus = any(stim in next_desc for stim in stimulus_events)
                if is_next_stimulus:
                    break

    if len(rts) == 0:
        if verbose:
            print("⚠️  No valid RTs found")
        return None

    # Aggregate RTs
    if method == 'mean':
        rt = np.mean(rts)
    elif method == 'median':
        rt = np.median(rts)
    elif method == 'first':
        rt = rts[0]
    else:
        rt = np.mean(rts)

    if verbose:
        print(f"Extracted {len(rts)} RTs: {rt:.3f}s ({method})")
        print(f"  Range: [{min(rts):.3f}, {max(rts):.3f}]s")

    return rt


def normalize_rt(rt, rt_min=0.2, rt_max=2.0):
    """
    Normalize RT to [0, 1] range

    Args:
        rt: Response time in seconds
        rt_min: Minimum expected RT (default: 0.2s)
        rt_max: Maximum expected RT (default: 2.0s)

    Returns:
        Normalized RT in [0, 1]
    """
    if rt is None:
        return 0.5  # Default to middle if unknown

    # Clip to expected range
    rt_clipped = np.clip(rt, rt_min, rt_max)

    # Normalize to [0, 1]
    rt_normalized = (rt_clipped - rt_min) / (rt_max - rt_min)

    return rt_normalized


# Test function
if __name__ == '__main__':
    print("RT Extractor Module")
    print("This module extracts response times from MNE Raw objects")
    print("\nUsage:")
    print("  from data.rt_extractor import extract_response_time, normalize_rt")
    print("  rt = extract_response_time(raw)")
    print("  rt_norm = normalize_rt(rt)")
