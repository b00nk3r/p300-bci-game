"""
Real-time Preprocessor & Classifier
====================================
Contains all preprocessing functions (identical to the offline training
pipeline) and a RealtimeClassifier class that loads the trained model
and runs the full pipeline on a single trial's EEG data.

The preprocessing sequence is:
    bandpass → bad-channel detection → CAR → epoch extraction →
    baseline correction → artifact rejection → feature extraction →
    LDA scoring → direction decision
"""

import numpy as np
from scipy.signal import butter, filtfilt

from bci_config import (
    SR, N_CHANNELS,
    BPF_LOW, BPF_HIGH, BPF_ORDER,
    EPOCH_PRE_MS, EPOCH_POST_MS,
    BASELINE_START_MS, BASELINE_END_MS,
    ARTIFACT_PP_THRESHOLD,
    BAD_CH_LOW_FACTOR, BAD_CH_HIGH_FACTOR,
    DEFAULT_WINDOW_START_MS, DEFAULT_WINDOW_END_MS,
    DEFAULT_DEC_WINDOW, DEFAULT_DEC_STEP,
    DIRECTION_MAP, DIRECTION_NAMES,
)


# =====================================================================
# Preprocessing functions (must be identical to offline utils.py)
# =====================================================================

def apply_bandpass_filter(eeg, low=BPF_LOW, high=BPF_HIGH, sr=SR, order=BPF_ORDER):
    """Zero-phase Butterworth bandpass on continuous EEG (n_samples, n_channels)."""
    b, a = butter(order, [low, high], btype="bandpass", fs=sr)
    filtered = filtfilt(b, a, eeg, axis=0)
    return filtered


def detect_bad_channels(eeg_filtered, low_factor=BAD_CH_LOW_FACTOR,
                        high_factor=BAD_CH_HIGH_FACTOR):
    """Detect channels with abnormally low/high std relative to median."""
    channel_stds = eeg_filtered.std(axis=0)
    median_std = np.median(channel_stds)

    bad_channels = []
    for ch in range(eeg_filtered.shape[1]):
        if channel_stds[ch] < low_factor * median_std:
            bad_channels.append(ch)
        elif channel_stds[ch] > high_factor * median_std:
            bad_channels.append(ch)

    return bad_channels, channel_stds


def apply_car(eeg, bad_channels):
    """Common Average Re-reference, excluding bad channels from the average."""
    good_channels = [
        ch for ch in range(eeg.shape[1]) if ch not in bad_channels
    ]
    avg = eeg[:, good_channels].mean(axis=1, keepdims=True)
    eeg_car = eeg - avg
    return eeg_car


def extract_epochs(eeg, flash_events, sr=SR,
                   pre_ms=EPOCH_PRE_MS, post_ms=EPOCH_POST_MS):
    """Cut epochs around each flash event from continuous EEG.

    Args:
        eeg: (n_samples, n_channels) – filtered & CAR'd continuous data.
        flash_events: list of dicts with "sample_idx" and "direction".

    Returns:
        epochs: (n_valid, epoch_len, n_channels)
        valid_events: corresponding event dicts
    """
    pre_samples = int(pre_ms * sr / 1000)
    post_samples = int(post_ms * sr / 1000)

    epochs = []
    valid_events = []

    for event in flash_events:
        idx = event["sample_idx"]
        start = idx - pre_samples
        end = idx + post_samples

        if start < 0 or end > eeg.shape[0]:
            continue

        epochs.append(eeg[start:end, :])
        valid_events.append(event)

    if not epochs:
        return np.empty((0, pre_samples + post_samples, eeg.shape[1]),
                         dtype=np.float32), valid_events

    return np.array(epochs, dtype=np.float32), valid_events


def apply_baseline_correction(epochs, sr=SR, pre_ms=EPOCH_PRE_MS,
                               baseline_start_ms=BASELINE_START_MS,
                               baseline_end_ms=BASELINE_END_MS):
    """Subtract the mean of the baseline window from each epoch."""
    pre_samples = int(pre_ms * sr / 1000)
    bl_start = pre_samples + int(baseline_start_ms * sr / 1000)
    bl_end = pre_samples + int(baseline_end_ms * sr / 1000)

    baseline_mean = epochs[:, bl_start:bl_end, :].mean(axis=1, keepdims=True)
    return epochs - baseline_mean


def reject_artifacts(epochs, pp_threshold=ARTIFACT_PP_THRESHOLD):
    """Reject epochs where peak-to-peak amplitude exceeds threshold."""
    n_epochs = epochs.shape[0]
    is_clean = np.ones(n_epochs, dtype=bool)

    for i in range(n_epochs):
        pp = epochs[i].max(axis=0) - epochs[i].min(axis=0)
        if pp.max() > pp_threshold:
            is_clean[i] = False

    return is_clean


# =====================================================================
# Feature extraction (must match training exactly)
# =====================================================================

def overlapping_moving_average(data, window_size, step_size):
    """Decimate 2-D data with overlapping moving-average windows."""
    n_time, n_channels = data.shape
    n_output = (n_time - window_size) // step_size + 1

    decimated = np.zeros((n_output, n_channels), dtype=data.dtype)
    for i in range(n_output):
        start = i * step_size
        end = start + window_size
        decimated[i] = data[start:end, :].mean(axis=0)

    return decimated


def extract_single_epoch_features(epoch, sr=SR, pre_ms=EPOCH_PRE_MS,
                                   window_start_ms=DEFAULT_WINDOW_START_MS,
                                   window_end_ms=DEFAULT_WINDOW_END_MS,
                                   dec_window=DEFAULT_DEC_WINDOW,
                                   dec_step=DEFAULT_DEC_STEP):
    """Extract the feature vector for one epoch (must match training)."""
    pre_samp = int(pre_ms * sr / 1000)
    start_idx = pre_samp + int(window_start_ms * sr / 1000)
    end_idx = pre_samp + int(window_end_ms * sr / 1000)

    trimmed = epoch[start_idx:end_idx, :]
    decimated = overlapping_moving_average(trimmed, dec_window, dec_step)
    return decimated.flatten()


# =====================================================================
# Trial-level decision
# =====================================================================

def predict_direction(scores, directions):
    """Pick the direction with the highest mean LDA score.

    Args:
        scores:     1-D array of LDA decision_function values (one per epoch).
        directions: 1-D array of direction codes (0-3), one per epoch.

    Returns:
        predicted_direction: int (0-3)
        direction_scores:    dict {direction_code: mean_score}
    """
    direction_scores = {}
    for d in range(4):
        mask = directions == d
        if mask.sum() > 0:
            direction_scores[d] = float(scores[mask].mean())
        else:
            direction_scores[d] = float("-inf")

    predicted_direction = max(direction_scores, key=direction_scores.get)
    return predicted_direction, direction_scores


# =====================================================================
# RealtimeClassifier — loads the model and runs the full pipeline
# =====================================================================

class RealtimeClassifier:
    """Wraps the trained LDA model and the full preprocessing pipeline."""

    def __init__(self, model_path: str):
        """Load the model artifact from a joblib file.

        The artifact dict is expected to contain at minimum:
            - "model": trained sklearn LDA object
            - "feature_params": dict with window_start_ms, window_end_ms,
                                dec_window, dec_step
        """
        import joblib

        artifact = joblib.load(model_path)
        self.model = artifact["model"]

        fp = artifact.get("feature_params", {})
        self.window_start_ms = fp.get("window_start_ms", DEFAULT_WINDOW_START_MS)
        self.window_end_ms = fp.get("window_end_ms", DEFAULT_WINDOW_END_MS)
        self.dec_window = fp.get("dec_window", DEFAULT_DEC_WINDOW)
        self.dec_step = fp.get("dec_step", DEFAULT_DEC_STEP)

        print(f"  Model loaded: {artifact.get('model_name', 'unknown')}")
        print(f"  Feature params: window={self.window_start_ms}-{self.window_end_ms}ms, "
              f"dec={self.dec_window}/{self.dec_step}")
        cv = artifact.get("cv_metrics", {})
        if cv:
            print(f"  CV metrics: epoch_auc={cv.get('epoch_auc', '?'):.3f}, "
                  f"trial_acc={cv.get('trial_acc', '?'):.3f}")

    # ------------------------------------------------------------------

    def classify_trial(self, eeg_chunk, chunk_start_time, flash_events,
                       bad_channels=None):
        """Run the full pipeline on one trial and return the predicted direction.

        Args:
            eeg_chunk:        (n_samples, n_channels) raw EEG from the buffer.
            chunk_start_time: LSL timestamp of the first sample in *eeg_chunk*.
            flash_events:     list of dicts with "time" (LSL ts) and "direction" (str).
            bad_channels:     list of channel indices to exclude from CAR.

        Returns:
            dict with keys:
                predicted_direction  – str ("up"/"down"/"left"/"right")
                direction_scores     – dict {direction_str: mean_score}
                n_clean_epochs       – int
                n_total_epochs       – int
        """
        if bad_channels is None:
            bad_channels = []

        # 1. Bandpass filter
        eeg_filtered = apply_bandpass_filter(eeg_chunk)

        # 2. If no bad channels provided, detect them on this chunk
        if not bad_channels:
            bad_channels, _ = detect_bad_channels(eeg_filtered)

        # 3. Common average re-reference
        eeg_car = apply_car(eeg_filtered, bad_channels)

        # 4. Compute per-event sample indices relative to chunk start
        events_with_idx = []
        for ev in flash_events:
            sample_idx = int((ev["time"] - chunk_start_time) * SR)
            events_with_idx.append({
                "sample_idx": sample_idx,
                "direction": ev["direction"],
            })

        # 5. Extract epochs
        epochs, valid_events = extract_epochs(eeg_car, events_with_idx)
        n_total = len(valid_events)

        if n_total == 0:
            return {
                "predicted_direction": None,
                "direction_scores": {},
                "n_clean_epochs": 0,
                "n_total_epochs": 0,
            }

        # 6. Baseline correction
        epochs = apply_baseline_correction(epochs)

        # 7. Artifact rejection
        is_clean = reject_artifacts(epochs)
        clean_epochs = epochs[is_clean]
        clean_events = [ev for ev, ok in zip(valid_events, is_clean) if ok]
        n_clean = len(clean_events)

        if n_clean == 0:
            return {
                "predicted_direction": None,
                "direction_scores": {},
                "n_clean_epochs": 0,
                "n_total_epochs": n_total,
            }

        # 8. Feature extraction
        X = np.array([
            extract_single_epoch_features(
                ep,
                window_start_ms=self.window_start_ms,
                window_end_ms=self.window_end_ms,
                dec_window=self.dec_window,
                dec_step=self.dec_step,
            )
            for ep in clean_epochs
        ], dtype=np.float64)

        # 9. LDA scoring
        scores = self.model.decision_function(X)

        # 10. Direction decision
        dir_codes = np.array(
            [DIRECTION_MAP[ev["direction"]] for ev in clean_events]
        )
        predicted_code, dir_scores = predict_direction(scores, dir_codes)
        predicted_name = DIRECTION_NAMES[predicted_code]

        dir_scores_named = {
            DIRECTION_NAMES[k]: v for k, v in dir_scores.items()
        }

        return {
            "predicted_direction": predicted_name,
            "direction_scores": dir_scores_named,
            "n_clean_epochs": n_clean,
            "n_total_epochs": n_total,
        }
