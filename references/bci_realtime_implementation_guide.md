# Real-Time BCI Pipeline — Implementation Guide

## What This Document Is

This document contains everything you need to implement a real-time Brain-Computer Interface (BCI) pipeline for an existing P300 BCI game. The game already exists and works. Your job is to add the modules that receive live EEG data, preprocess it, classify it with a pre-trained model, and feed the result back into the game to control the character.

Read this entire document before writing any code.

---

## 1. Project Context

### What the game does

The P300 BCI Game is a Pygame application where a user controls a maze character using brain signals. Four directional arrows (Up, Down, Left, Right) flash one at a time on screen. When the user mentally focuses on one arrow, their brain produces a P300 event-related potential each time that arrow flashes. A classifier detects which direction produced the P300 and moves the character.

### How a trial works

1. User presses SPACE to start a trial.
2. The game flashes all 4 arrows in random order — this is one "sequence." Default is 15 sequences per trial. Each sequence flashes all 4 arrows once, so 15 sequences = 60 total flashes.
3. Each flash lasts 100ms, with 125ms inter-stimulus interval (225ms between flash onsets).
4. After all flashes complete, the game enters PROCESSING state and waits for a direction selection.
5. Currently, the user presses keys 1-4 to simulate BCI selection. You are replacing this with real classification.

### How the game state machine works

The `ArrowManager` class in `src/stimulus/arrow_manager.py` manages a `SelectionState` enum:
- `IDLE` → user presses SPACE → `FLASHING`
- `FLASHING` → all sequences done → `PROCESSING`
- `PROCESSING` → direction received → `FEEDBACK`
- `FEEDBACK` → animation done → `COMPLETE`
- `COMPLETE` → reset → `IDLE`

During `PROCESSING`, the game currently waits for keyboard input (keys 1-4 mapped to UP/DOWN/LEFT/RIGHT via `simulate_selection(direction)`). Your real-time classifier will provide the direction instead.

### How the game records flash events

The `TriggerManager` class (defined inside `src/stimulus/arrow_manager.py`) records each flash with:
- A trigger code (10=UP, 11=DOWN, 12=LEFT, 13=RIGHT)
- A timestamp (from `time.perf_counter()`)
- The current target direction (during calibration)

It writes these to text files in `data/sessions/`.

The `TimingController` class in `src/stimulus/timing_controller.py` pre-schedules all flash events and uses `time.perf_counter()` for sub-millisecond timing.

### EEG Hardware

The EEG device is a g.tec g.Nautilus RESEARCH wireless headset with 16 channels, sampling at 500 Hz. It transmits wirelessly to a Base Station connected to the PC via USB. The recording software is g.Recorder, which can output an LSL (Lab Streaming Layer) stream.

---

## 2. The Trained ML Model

### What model we're using

A Linear Discriminant Analysis (LDA) classifier trained on single-trial EEG epochs. The model is saved as a joblib file containing a dictionary (called "artifact") with the model and all parameters needed to reproduce the feature extraction.

### The artifact structure

```python
artifact = {
    "model_name": "single_trial_lda",
    "model": <sklearn LDA object>,        # the trained model
    "model_params": {
        "solver": "lsqr",
        "shrinkage": "auto",
    },
    "feature_params": {
        "window_start_ms": <int>,          # e.g. 0
        "window_end_ms": <int>,            # e.g. 800
        "dec_window": <int>,               # e.g. 20
        "dec_step": <int>,                 # e.g. 10
    },
    "cv_metrics": {
        "epoch_auc": <float>,
        "trial_acc": <float>,
    },
    # ... plus cross-validation arrays not needed at runtime
}
```

### How classification works at trial level

For each trial (60 flashes, 15 per direction):
1. Each flash produces one EEG epoch (a short window of EEG data around the flash)
2. Each epoch is preprocessed and converted to a feature vector
3. The LDA model scores each feature vector with `decision_function()` — higher score = more likely to be a P300 (target)
4. Scores are grouped by direction (15 scores per direction)
5. The mean score per direction is computed
6. The direction with the highest mean score is the predicted direction

---

## 3. The Offline Preprocessing Pipeline (Must Be Replicated Exactly)

The real-time pipeline MUST match the offline training pipeline exactly, or the model's accuracy will degrade. Here is every step of the offline pipeline and the exact code.

### 3.1 Bandpass Filter

Applied to continuous EEG data (not epochs). Butterworth bandpass, 0.5–30 Hz, order 4, zero-phase (filtfilt).

```python
from scipy.signal import butter, filtfilt

def apply_bandpass_filter(eeg, low=0.5, high=30.0, sr=500, order=4):
    """
    Apply zero-phase bandpass filter to continuous EEG.
    
    Args:
        eeg: numpy array of shape (n_samples, n_channels)
        low: low cutoff frequency in Hz
        high: high cutoff frequency in Hz
        sr: sampling rate in Hz
        order: filter order
    
    Returns:
        filtered: numpy array of same shape as eeg
    """
    b, a = butter(order, [low, high], btype="bandpass", fs=sr)
    filtered = filtfilt(b, a, eeg, axis=0)
    return filtered
```

**CRITICAL**: This uses `filtfilt` (zero-phase), which requires the full data segment. In real time, this is fine because we filter the entire trial's EEG chunk at once (not sample-by-sample streaming).

### 3.2 Bad Channel Detection

Detects channels with abnormally low or high standard deviation relative to median.

```python
import numpy as np

def detect_bad_channels(eeg_filtered, low_factor=0.01, high_factor=4.0):
    """
    Detect bad EEG channels based on standard deviation.
    
    Args:
        eeg_filtered: numpy array of shape (n_samples, n_channels), already bandpass filtered
        low_factor: channels with std < low_factor * median_std are bad (flat/dead)
        high_factor: channels with std > high_factor * median_std are bad (noisy)
    
    Returns:
        bad_channels: list of channel indices (0-based)
        channel_stds: numpy array of per-channel standard deviations
    """
    channel_stds = eeg_filtered.std(axis=0)
    median_std = np.median(channel_stds)

    bad_channels = []
    for ch in range(eeg_filtered.shape[1]):
        if channel_stds[ch] < low_factor * median_std:
            bad_channels.append(ch)
        elif channel_stds[ch] > high_factor * median_std:
            bad_channels.append(ch)

    return bad_channels, channel_stds
```

### 3.3 Common Average Re-referencing (CAR)

Subtracts the mean of good channels from all channels.

```python
def apply_car(eeg, bad_channels):
    """
    Apply Common Average Re-referencing, excluding bad channels from the average.
    
    Args:
        eeg: numpy array of shape (n_samples, n_channels)
        bad_channels: list of channel indices to exclude from average
    
    Returns:
        eeg_car: re-referenced numpy array of same shape
    """
    good_channels = [ch for ch in range(eeg.shape[1]) if ch not in bad_channels]
    avg = eeg[:, good_channels].mean(axis=1, keepdims=True)
    eeg_car = eeg - avg
    return eeg_car
```

### 3.4 Epoch Extraction

Cut a time window around each flash event from the continuous EEG.

- Pre-stimulus: 200ms before flash onset
- Post-stimulus: 800ms after flash onset
- Total epoch length: 1000ms = 500 samples at 500 Hz
- Epoch shape: (500, 16) — 500 time points × 16 channels

```python
def extract_epochs(eeg, flash_events, sr=500, pre_ms=200, post_ms=800):
    """
    Extract epochs from continuous EEG around flash event timestamps.
    
    Args:
        eeg: numpy array of shape (n_samples, n_channels), filtered and CAR'd
        flash_events: list of dicts with at least:
            - "sample_idx": integer sample index into eeg where the flash occurred
            - "direction": string, one of "up", "down", "left", "right"
        sr: sampling rate
        pre_ms: ms before flash onset to include
        post_ms: ms after flash onset to include
    
    Returns:
        epochs: numpy array of shape (n_valid_epochs, epoch_length, n_channels)
        valid_events: list of events that produced valid (non-boundary) epochs
    """
    pre_samples = int(pre_ms * sr / 1000)   # 100 samples
    post_samples = int(post_ms * sr / 1000)  # 400 samples
    # Total epoch length = pre_samples + post_samples = 500 samples

    epochs = []
    valid_events = []

    for event in flash_events:
        idx = event["sample_idx"]
        start = idx - pre_samples
        end = idx + post_samples

        if start < 0 or end > eeg.shape[0]:
            continue

        epoch = eeg[start:end, :]
        epochs.append(epoch)
        valid_events.append(event)

    return np.array(epochs, dtype=np.float32), valid_events
```

### 3.5 Baseline Correction

Subtract the mean of the baseline window from each epoch. Baseline window is -100ms to 0ms relative to flash onset.

```python
def apply_baseline_correction(epochs, sr=500, pre_ms=200,
                               baseline_start_ms=-100, baseline_end_ms=0):
    """
    Apply baseline correction to epochs.
    
    Args:
        epochs: numpy array of shape (n_epochs, epoch_length, n_channels)
        sr: sampling rate
        pre_ms: pre-stimulus time in the epoch (how much before stimulus onset)
        baseline_start_ms: start of baseline window relative to stimulus onset (negative = before)
        baseline_end_ms: end of baseline window relative to stimulus onset
    
    Returns:
        corrected: baseline-corrected epochs, same shape
    """
    pre_samples = int(pre_ms * sr / 1000)  # 100 samples (200ms at 500Hz)

    # Convert baseline window to sample indices within the epoch
    bl_start = pre_samples + int(baseline_start_ms * sr / 1000)  # 100 + (-50) = 50
    bl_end = pre_samples + int(baseline_end_ms * sr / 1000)      # 100 + 0 = 100

    baseline_mean = epochs[:, bl_start:bl_end, :].mean(axis=1, keepdims=True)
    return epochs - baseline_mean
```

### 3.6 Artifact Rejection

Reject epochs where peak-to-peak amplitude exceeds threshold on any channel.

```python
def reject_artifacts(epochs, pp_threshold=150.0):
    """
    Reject epochs with excessive peak-to-peak amplitude.
    
    Args:
        epochs: numpy array of shape (n_epochs, epoch_length, n_channels)
        pp_threshold: maximum allowed peak-to-peak amplitude in µV
    
    Returns:
        is_clean: boolean array of shape (n_epochs,)
    """
    n_epochs = epochs.shape[0]
    is_clean = np.ones(n_epochs, dtype=bool)

    for i in range(n_epochs):
        pp = epochs[i].max(axis=0) - epochs[i].min(axis=0)
        if pp.max() > pp_threshold:
            is_clean[i] = False

    return is_clean
```

### 3.7 Feature Extraction

This is the most critical part to match exactly. The model was trained on features extracted this way:

1. **Trim** the epoch to the analysis window (e.g., 0ms to 800ms post-stimulus). Since the epoch starts at -200ms (pre_ms=200), the 0ms mark is at sample index 100 (200ms × 500Hz / 1000).
2. **Decimate** using overlapping moving average with a specific window and step size.
3. **Flatten** the resulting 2D array into a 1D feature vector.

```python
def overlapping_moving_average(data, window_size, step_size):
    """
    Decimate 2D data using overlapping moving average windows.
    
    Args:
        data: numpy array of shape (n_time, n_channels)
        window_size: number of samples per averaging window
        step_size: number of samples to advance between windows
    
    Returns:
        decimated: numpy array of shape (n_output, n_channels)
    """
    n_time, n_channels = data.shape
    n_output = (n_time - window_size) // step_size + 1

    decimated = np.zeros((n_output, n_channels), dtype=data.dtype)
    for i in range(n_output):
        start = i * step_size
        end = start + window_size
        decimated[i] = data[start:end, :].mean(axis=0)

    return decimated


def extract_single_epoch_features(epoch, sr=500, pre_ms=200,
                                   window_start_ms=0, window_end_ms=800,
                                   dec_window=20, dec_step=10):
    """
    Extract features from a single epoch for real-time classification.
    
    Args:
        epoch: numpy array of shape (epoch_length, n_channels), e.g. (500, 16)
        sr: sampling rate
        pre_ms: how much pre-stimulus time is in the epoch
        window_start_ms: analysis window start relative to stimulus onset
        window_end_ms: analysis window end relative to stimulus onset
        dec_window: decimation window size in samples
        dec_step: decimation step size in samples
    
    Returns:
        features: 1D numpy array (the feature vector for this epoch)
    """
    pre_samp = int(pre_ms * sr / 1000)
    start_idx = pre_samp + int(window_start_ms * sr / 1000)
    end_idx = pre_samp + int(window_end_ms * sr / 1000)

    trimmed = epoch[start_idx:end_idx, :]
    decimated = overlapping_moving_average(trimmed, dec_window, dec_step)
    return decimated.flatten()
```

### 3.8 Trial-Level Decision

After getting a score for each epoch, group by direction and pick the winner:

```python
def predict_direction(scores, directions):
    """
    Predict attended direction from single-epoch LDA scores.
    
    Args:
        scores: 1D numpy array of LDA decision_function scores, one per epoch
        directions: 1D numpy array of direction codes (0=up, 1=down, 2=left, 3=right),
                    one per epoch, indicating which arrow was flashing for that epoch
    
    Returns:
        predicted_direction: int (0-3)
        direction_scores: dict mapping direction code to mean score
    """
    direction_scores = {}
    for d in range(4):
        mask = directions == d
        if mask.sum() > 0:
            direction_scores[d] = scores[mask].mean()
        else:
            direction_scores[d] = float('-inf')

    predicted_direction = max(direction_scores, key=direction_scores.get)
    return predicted_direction, direction_scores
```

---

## 4. Files to Create

### 4.1 `bci_config.py` (new file in project root)

Contains all parameters for the real-time BCI pipeline. These must match the training pipeline.

```python
"""
Real-time BCI configuration.
All preprocessing parameters MUST match the offline training pipeline.
"""

# --- EEG Hardware ---
SR = 500                    # Sampling rate in Hz
N_CHANNELS = 16             # Number of EEG channels

# --- Preprocessing (must match training) ---
BPF_LOW = 0.5               # Bandpass filter low cutoff (Hz)
BPF_HIGH = 30.0             # Bandpass filter high cutoff (Hz)
BPF_ORDER = 4               # Butterworth filter order

BASELINE_START_MS = -100     # Baseline correction window start (ms relative to stimulus)
BASELINE_END_MS = 0          # Baseline correction window end

ARTIFACT_PP_THRESHOLD = 150.0  # Peak-to-peak rejection threshold (µV)

BAD_CH_LOW_FACTOR = 0.01    # Bad channel detection: low std threshold
BAD_CH_HIGH_FACTOR = 4.0    # Bad channel detection: high std threshold

# --- Epoching ---
EPOCH_PRE_MS = 200           # Pre-stimulus epoch window (ms)
EPOCH_POST_MS = 800          # Post-stimulus epoch window (ms)

# --- Feature Extraction (loaded from model artifact at runtime) ---
# These are defaults; the actual values come from the saved model file
DEFAULT_WINDOW_START_MS = 0
DEFAULT_WINDOW_END_MS = 800
DEFAULT_DEC_WINDOW = 20
DEFAULT_DEC_STEP = 10

# --- Real-time Buffer ---
BUFFER_DURATION_S = 60       # Circular buffer duration in seconds

# --- Model ---
MODEL_PATH = "models/single_trial_lda_best_model.joblib"

# --- Direction Mapping ---
DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
}
DIRECTION_NAMES = {v: k for k, v in DIRECTION_MAP.items()}
```

### 4.2 `src/communication/eeg_receiver.py` (new file)

Responsibilities:
- Connect to the LSL EEG stream from g.Recorder
- Run a background daemon thread that continuously reads samples into a circular buffer
- Provide a thread-safe method to retrieve a chunk of EEG data between two LSL timestamps
- Provide the current LSL clock time (for the game to timestamp flash events)

Implementation details:

**Circular buffer**: A numpy array of shape `(buffer_size, n_channels)` where `buffer_size = SR * BUFFER_DURATION_S`. A parallel array of shape `(buffer_size,)` stores the LSL timestamp for each sample. An integer `write_pos` tracks where to write next (wraps around).

**Background thread**: Runs in a `while self._running` loop. Each iteration calls `inlet.pull_chunk(timeout=0.1)` which returns a batch of samples and timestamps. Write them into the circular buffer at `write_pos`, advance `write_pos` modulo `buffer_size`. Use a `threading.Lock` around all buffer writes and reads.

**`get_chunk(start_time, end_time)` method**: Given two LSL timestamps, find the samples in the buffer that fall within that range. Return a numpy array of shape `(n_samples_in_range, n_channels)` and the corresponding timestamps array. Use the lock when reading. Handle the circular wrap-around case (where the data spans the buffer boundary).

**`get_current_time()` method**: Returns `pylsl.local_clock()`. The game should call this instead of `time.perf_counter()` when recording flash timestamps, so that flash timestamps and EEG timestamps are in the same clock domain.

**Connection handling**: The `start()` method should call `pylsl.resolve_stream('type', 'EEG')` to find the stream, create a `StreamInlet`, then start the background thread. If no stream is found, raise a clear error. The `stop()` method sets `self._running = False` and joins the thread.

**Important**: The LSL inlet should be created with `max_buflen=BUFFER_DURATION_S` to let LSL buffer enough data. Also set `max_chunklen=0` for sample-by-sample delivery or a small chunk length.

**Graceful degradation**: Add a `is_connected()` method that returns whether the receiver is actively receiving data. The game can check this and fall back to keyboard input if EEG is not available.

### 4.3 `src/communication/realtime_preprocessor.py` (new file)

Responsibilities:
- Contain all preprocessing and feature extraction functions
- Load the trained model from the joblib artifact
- Provide a single method that takes raw EEG + flash events and returns a predicted direction

This file should contain:
1. All the preprocessing functions from Section 3 of this document (bandpass filter, CAR, epoch extraction, baseline correction, artifact rejection, feature extraction, overlapping moving average)
2. A `RealtimeClassifier` class that:
   - Loads the model artifact in `__init__`
   - Reads feature_params from the artifact
   - Provides `classify_trial(eeg_chunk, chunk_start_time, flash_events, bad_channels)` that runs the full pipeline and returns the predicted direction

The `classify_trial` method flow:
1. Apply bandpass filter to the raw EEG chunk
2. Apply CAR (using provided bad_channels list)
3. For each flash event, compute the sample index: `int((flash_time - chunk_start_time) * SR)`
4. Extract epochs using those sample indices
5. Apply baseline correction
6. Reject artifact epochs
7. For each clean epoch, extract features using the model's feature_params
8. Stack features into a matrix and call `model.decision_function(X)`
9. Group scores by direction, compute mean per direction, argmax → return predicted direction and confidence info

**What flash_events look like**: A list of dicts, each with:
- `"time"`: LSL timestamp (float) of when the flash occurred
- `"direction"`: string, one of "up", "down", "left", "right"

**Return value**: A dict with:
- `"predicted_direction"`: string, one of "up", "down", "left", "right"
- `"direction_scores"`: dict mapping direction string to mean score
- `"n_clean_epochs"`: int, how many epochs survived artifact rejection
- `"n_total_epochs"`: int, total epochs before rejection

### 4.4 `src/communication/bci_controller.py` (new file)

Responsibilities:
- High-level coordinator between EEGReceiver, RealtimeClassifier, and the game
- Manage the flash event collection during a trial
- Handle bad channel detection (calibration)
- Provide simple interface for the game to use

The class should have these methods:

**`__init__(model_path)`**: Creates EEGReceiver and RealtimeClassifier instances. Sets `bad_channels = []` initially.

**`start()`**: Starts the EEG receiver. May block briefly while connecting to LSL.

**`stop()`**: Stops the EEG receiver.

**`is_connected()`**: Returns whether EEG is being received.

**`begin_trial()`**: Called when a trial starts (SPACE pressed). Clears the flash event list. Records the trial start time.

**`record_flash(direction)`**: Called each time an arrow flashes during the trial. Records `{"time": eeg_receiver.get_current_time(), "direction": direction}` into the flash event list.

**`end_trial()`**: Called when all flashes are done (SelectionState goes to PROCESSING). This is the main method. It:
1. Determines the time range needed: `start = earliest_flash_time - EPOCH_PRE_MS/1000 - 0.5` (extra 0.5s margin for filter edge effects), `end = latest_flash_time + EPOCH_POST_MS/1000 + 0.5`
2. Retrieves the EEG chunk from the receiver
3. Calls `classifier.classify_trial(...)` with the chunk and flash events
4. Returns the result

**`run_calibration(duration_s=10)`**: Optional method. Collects `duration_s` seconds of resting EEG, runs bad channel detection, and stores the result. Can be called at session start before gameplay begins.

### 4.5 `src/communication/__init__.py` (modify existing)

Currently this file is a placeholder with commented-out exports. Update it to export the new classes:

```python
from .bci_controller import BCIController
```

---

## 5. Integration With Existing Game Code

### 5.1 Changes to `src/stimulus/arrow_manager.py`

The `ArrowManager` class needs minimal modifications:

**Add a reference to BCIController**: The `ArrowManager` should optionally accept a `bci_controller` parameter. If provided, it uses real BCI; if not, it falls back to keyboard simulation (current behavior).

**During FLASHING state**: Every time a flash event fires (in the `update()` method or wherever flash events are processed), if `bci_controller` is not None, call `bci_controller.record_flash(direction_string)`. The direction string should be "up", "down", "left", or "right".

**On trial start**: When `start_selection()` is called, if `bci_controller` is not None, also call `bci_controller.begin_trial()`.

**On transition to PROCESSING**: When the state changes from FLASHING to PROCESSING, if `bci_controller` is not None:
1. Call `result = bci_controller.end_trial()`
2. Extract the predicted direction from the result
3. Convert the direction string to a Direction enum value
4. Call `self.simulate_selection(direction)` with that value

This means the PROCESSING state becomes nearly instantaneous when using real BCI — it doesn't wait for user input, it immediately classifies and moves on.

### 5.2 Changes to `main.py`

**During initialization** (in the `Application.__init__` method or initialization sequence):

```python
# After existing initialization...
self.bci_controller = None  # Default: no BCI

# Try to initialize BCI if model exists
try:
    from src.communication import BCIController
    import os
    model_path = "models/single_trial_lda_best_model.joblib"
    if os.path.exists(model_path):
        self.bci_controller = BCIController(model_path)
        self.bci_controller.start()
        print("BCI controller initialized and connected to EEG stream")
    else:
        print(f"No model found at {model_path}, running in keyboard mode")
except ImportError:
    print("pylsl not installed, running in keyboard mode")
except Exception as e:
    print(f"BCI initialization failed: {e}, running in keyboard mode")
```

**Pass bci_controller to ArrowManager**: When creating the ArrowManager, pass the bci_controller reference so it can use it during trials.

**During shutdown** (ESC handler or quit):

```python
if self.bci_controller:
    self.bci_controller.stop()
```

### 5.3 Changes to calibration system

The calibration system in `main.py` also records flash events. During calibration, the BCI controller should also record flashes (for potential real-time feedback), but the calibration itself doesn't need real-time classification — it's just collecting labeled data. So during calibration mode, you should still call `bci_controller.record_flash()` for each flash, but you don't need to call `end_trial()` at the end of each calibration phase. Instead, you could optionally run bad channel detection after the first calibration block.

---

## 6. Dependencies to Add

Add to `requirements.txt`:

```
pylsl>=1.16.0
joblib>=1.3.0
```

The game already lists pylsl as optional. scipy, numpy, and sklearn should already be available (scipy and numpy are in existing requirements; sklearn is needed for the model).

If sklearn is not in the existing requirements, add:

```
scikit-learn>=1.3.0
```

---

## 7. Directory Setup

Create the `models/` directory in the project root. The trained model file (`single_trial_lda_best_model.joblib`) will be placed there by the user. Your code should handle the case where this file does not exist (fall back to keyboard mode).

---

## 8. Important Technical Details

### Timestamp synchronization

The game currently uses `time.perf_counter()` for flash timing. For real-time BCI, flash events must be timestamped with `pylsl.local_clock()` instead, because the EEG data from the LSL inlet uses LSL's clock. Both clocks run at the same rate on the same machine, but they have different epoch origins. You must use the same clock for both.

The cleanest approach: when recording a flash event for BCI, call `pylsl.local_clock()` at that moment. The `TimingController` can continue using `time.perf_counter()` for its internal timing logic — that doesn't need to change. You only need the LSL timestamp for the BCI pipeline.

### Filter edge effects

When you apply `filtfilt` to a short chunk of data, the edges can have artifacts. The 0.5-second margin added in `end_trial()` when requesting the EEG chunk accounts for this. Make sure you request more data than you strictly need, then epoch from the middle of the filtered chunk.

### Thread safety

The `EEGReceiver` runs a background thread. The game runs on the main thread. The `get_chunk()` method must be thread-safe (use a lock). The classification in `end_trial()` happens on the main thread and may take 100-300ms. This is acceptable because it happens during the PROCESSING state when the game is just waiting anyway.

### What if LSL is not available

The system should gracefully degrade. If pylsl is not installed, or if no LSL stream is found, or if the model file doesn't exist, the game should continue to work in keyboard-simulation mode exactly as it does today. Never crash due to BCI initialization failure.

### Number of sequences is variable

The number of sequences per trial is configurable (in the game's config.py TimingConfig, default 15). The real-time pipeline doesn't need to know this number in advance — it just processes however many flash events were recorded during the trial. More sequences = more epochs per direction = more reliable averaging = better accuracy.

---

## 9. Testing Strategy

### Test without EEG hardware

For development and testing without the actual EEG headset:

1. Create a mock LSL stream that sends random or pre-recorded EEG data. A simple script can do this:

```python
"""mock_eeg_stream.py — Run this to simulate an EEG stream for testing"""
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet

info = StreamInfo('MockEEG', 'EEG', 16, 500, 'float32', 'mock_eeg_001')
outlet = StreamOutlet(info)

print("Mock EEG stream started. Press Ctrl+C to stop.")
sample_interval = 1.0 / 500
while True:
    sample = (np.random.randn(16) * 10).tolist()  # random noise, ~10 µV
    outlet.push_sample(sample)
    time.sleep(sample_interval)
```

2. With this running, the game should successfully connect to the LSL stream. Classification results will be random (since the data is noise), but this tests the entire pipeline end-to-end.

### Test the preprocessor independently

Write a simple test that:
1. Creates synthetic EEG data with a known P300-like signal in one direction
2. Runs it through the full preprocessing pipeline
3. Verifies the feature vectors have the expected shape

### Test the circular buffer

Verify that `get_chunk()` correctly handles:
- Normal case (data within buffer)
- Wrap-around case (data spans the buffer boundary)
- Edge case (requested time range is partially outside buffer)

---

## 10. File Summary

New files to create:
1. `bci_config.py` — configuration parameters (project root)
2. `src/communication/eeg_receiver.py` — LSL inlet and circular buffer
3. `src/communication/realtime_preprocessor.py` — preprocessing + classification
4. `src/communication/bci_controller.py` — high-level coordinator
5. `models/.gitkeep` — placeholder for model directory

Files to modify:
1. `src/communication/__init__.py` — update exports
2. `src/stimulus/arrow_manager.py` — add BCI flash recording and auto-classification
3. `main.py` — initialize BCI controller, pass to ArrowManager, cleanup on exit
4. `requirements.txt` — add pylsl, joblib, scikit-learn

Optional file to create for testing:
1. `mock_eeg_stream.py` — fake LSL stream for development without hardware

---

## 11. Implementation Order

Follow this order. Each step should be independently testable before moving to the next.

1. **`bci_config.py`** — Just create the config file. No dependencies.
2. **`src/communication/eeg_receiver.py`** — Test standalone with mock_eeg_stream.py.
3. **`src/communication/realtime_preprocessor.py`** — Test standalone with synthetic data.
4. **`src/communication/bci_controller.py`** — Test standalone by manually providing flash events.
5. **Modify `src/stimulus/arrow_manager.py`** — Add BCI integration hooks.
6. **Modify `main.py`** — Wire everything together.
7. **Modify `src/communication/__init__.py`** — Update exports.
8. **Update `requirements.txt`** — Add new dependencies.
