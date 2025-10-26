# BREAKTHROUGH INSIGHT: Why We're Stuck at 1.0-1.5 NRMSE

## The Fundamental Problem

**Current Approach (WRONG):**
```
Recording (5 minutes, ~30 trials) → Average/Flatten → Predict ONE RT → NRMSE: 1.0-1.5
```

**What We Should Do (CORRECT):**
```
Recording → Extract 30 individual trials → Predict 30 RTs → Average predictions → NRMSE: ???
```

## Why This Matters

### Challenge 1 Task Structure:
- **Input**: EEG recordings with multiple trials of Contrast Change Detection task
- **Each Trial**: Stimulus onset → Participant responds → RT measured
- **Target**: Predict reaction time (RT) for EACH trial, not per recording

### What We're Doing Wrong:
1. Taking entire recording (5 min, 30,000 time points)
2. Truncating/padding to 200 time points (losing 99% of data!)
3. Predicting ONE reaction time per recording
4. This is like predicting average height of a class by looking at one blurry group photo

### What We Should Do:
1. Extract individual trials (stimulus onset - 0.5s to + 1.5s)
2. Each trial = ~200 time points @ 100Hz
3. Predict RT for EACH trial
4. Aggregate predictions per recording

## Key Insights from Neuroscience

### Reaction Time Depends On:

1. **Pre-Stimulus Alpha Power** (8-12 Hz)
   - High alpha = low attention = slower RT
   - Measured -500ms before stimulus

2. **Readiness Potential** (motor preparation)
   - Negative slow wave over motor cortex
   - Starts ~1s before movement

3. **P300 Component** (decision making)
   - Peak latency correlates with RT
   - ~300-500ms after stimulus

4. **Motor Beta Suppression**
   - Beta power drop = motor execution
   - Timing predicts RT

### Why Averaging Destroys Signal:
- Pre-stimulus alpha varies BETWEEN trials
- P300 latency varies BETWEEN trials
- Averaging 30 trials = blur all temporal information
- Like predicting when 30 people will sneeze by averaging their nose movements

## The Winning Strategy

### Trial-Level Architecture:

```
For each recording:
  1. Find all stimulus onset events
  2. For each trial:
     a. Extract [-500ms, +1500ms] around stimulus
     b. Extract features:
        - Pre-stimulus alpha power (attention state)
        - Early ERP components (N1, P1)
        - P300 latency and amplitude
        - Motor preparation (beta suppression)
        - Baseline RT tendency (slow vs fast subject)
     c. Predict RT for THIS trial
  3. Aggregate predictions (mean or median)
  4. Return recording-level prediction
```

### Why This Will Work:

1. **More Data**: 30 trials/recording × 767 recordings = 23,000+ training samples
2. **Temporal Precision**: Each trial preserves timing relationships
3. **Individual Variability**: Captures fast vs slow trials within same subject
4. **Neuroscience-Grounded**: Uses known RT predictors

### Expected Performance:

- **Current**: NRMSE 1.0-1.5 (random guessing with subject bias)
- **Trial-level**: NRMSE 0.7-0.9 (capturing trial variability)
- **Top team (0.976)**: Likely using trial-level or similar approach

## Implementation Plan

### Step 1: Extract Trials (High Priority!)
```python
def extract_trials_from_recording(raw, pre_stim=0.5, post_stim=1.5):
    """
    Extract individual trials from recording

    Returns:
        trials: List of (eeg_data, rt) tuples
        - eeg_data: (129 channels, 200 time points)
        - rt: reaction time for this trial
    """
    # Find stimulus onset events
    events = find_stimulus_events(raw)

    trials = []
    for event in events:
        # Extract [-500ms, +1500ms] around stimulus
        trial_data = raw.get_data(
            start=event['onset'] - pre_stim,
            stop=event['onset'] + post_stim
        )

        # Get RT for this specific trial
        rt = event['rt']

        trials.append((trial_data, rt))

    return trials
```

### Step 2: Trial-Level Model
```python
class TrialLevelRTPredictor(nn.Module):
    """
    Predicts RT from single trial
    Input: (batch, 129, 200) - ONE trial
    Output: (batch, 1) - RT for this trial
    """
    def __init__(self):
        # Spatial attention (which channels matter)
        self.spatial_attention = ChannelAttention(129)

        # Temporal CNN (extract ERPs, alpha, beta)
        self.temporal_cnn = TemporalCNN()

        # Pre-stimulus branch (attention state)
        self.pre_stim_branch = PreStimulusEncoder()

        # Post-stimulus branch (ERP, motor)
        self.post_stim_branch = PostStimulusEncoder()

        # RT predictor
        self.rt_head = nn.Linear(512, 1)
```

### Step 3: Aggregation Strategy
```python
def predict_recording_rt(model, recording):
    """
    Predict RT for entire recording by aggregating trial predictions
    """
    # Extract all trials
    trials = extract_trials_from_recording(recording)

    # Predict RT for each trial
    trial_rts = []
    for trial_data, true_rt in trials:
        predicted_rt = model(trial_data)
        trial_rts.append(predicted_rt)

    # Aggregate (median more robust than mean)
    recording_rt = torch.median(torch.stack(trial_rts))

    return recording_rt
```

## Why Previous Approaches Failed

1. **Domain Adaptation**: Tried to align subject distributions, but ignored trial structure
2. **Cross-Task Transfer**: Pre-trained on other tasks, but still averaged trials
3. **Hybrid CNN-Transformer**: Complex architecture, but wrong data granularity

All used **recording-level** prediction when task requires **trial-level** understanding!

## Critical Next Steps

1. ✅ Identify the flaw (DONE)
2. 🔄 Extract trial-level data structure
3. Build trial-level predictor
4. Test on validation set
5. Compare to current 1.0-1.5 baseline

**Expected Result**: NRMSE drops to 0.8-0.95 range (60-80% reduction in error)

---

This is the breakthrough we need. The question isn't "what complex architecture should we use?"

The question is: **"Are we predicting the right thing at the right granularity?"**

Answer: No. We need trial-level predictions, not recording-level predictions.
