"""
Real-Time P300 Classifier for the BCI Game
===========================================

Loads one trained model (.joblib) and classifies the attended direction
in real time.

Architecture:
    - A background thread continuously pulls EEG samples from an LSL
      stream and stores them in a thread-safe ring buffer.
    - When the game finishes flashing (trial complete), the main thread
      calls classify_trial() with the recorded flash events.
    - classify_trial() extracts epochs, preprocesses them to match the
      offline pipeline, runs the model, and returns the predicted
      direction.

Interface:
    classifier = RealtimeClassifier(model_path="models/10trials_model.joblib")
    classifier.start()
    classifier.record_flash(direction, timestamp)
    result = classifier.classify_trial()
    classifier.stop()
"""

import threading
import time
import numpy as np
import joblib
from scipy.signal import butter, filtfilt

try:
    from pylsl import StreamInlet, resolve_byprop, local_clock
except ImportError:
    raise ImportError(
        "pylsl is required for real-time EEG streaming. "
        "Install it with: pip install pylsl"
    )


# ── Constants matching the offline pipeline ──────────────────────────

from config import EEG_SAMPLING_RATE_HZ as SR, EEG_N_CHANNELS as N_CHANNELS

BPF_LOW = 0.5
BPF_HIGH = 30.0
BPF_ORDER = 4

EPOCH_PRE_MS = 200
EPOCH_POST_MS = 800
EPOCH_SAMPLES = int((EPOCH_PRE_MS + EPOCH_POST_MS) * SR / 1000)

BASELINE_START_MS = -100
BASELINE_END_MS = 0

ARTIFACT_PP_THRESHOLD = 150.0

BAD_CH_LOW_FACTOR = 0.01
BAD_CH_HIGH_FACTOR = 4.0

BUFFER_DURATION_S = 60
BUFFER_SAMPLES = BUFFER_DURATION_S * SR
EVENT_ALIGNMENT_TOLERANCE_S = 0.010

DIRECTION_MAP = {"up": 0, "down": 1, "left": 2, "right": 3}
DIRECTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

# Channels to drop for the "12ch no artifact" subset
# FP1=0, FP2=1, T7=5, T8=9  →  keep the rest
CH_NO_ARTIFACT = [2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15]


def _nearest_sample_index(
    timestamps, event_timestamp, tolerance_s=EVENT_ALIGNMENT_TOLERANCE_S
):
    """Return the closest sample index to an event timestamp."""
    if len(timestamps) == 0:
        return None

    insert_idx = np.searchsorted(timestamps, event_timestamp)
    best_idx = min(insert_idx, len(timestamps) - 1)

    if insert_idx > 0:
        prev_idx = insert_idx - 1
        if insert_idx >= len(timestamps) or (
            abs(timestamps[prev_idx] - event_timestamp)
            < abs(timestamps[best_idx] - event_timestamp)
        ):
            best_idx = prev_idx

    if abs(timestamps[best_idx] - event_timestamp) > tolerance_s:
        return None

    return int(best_idx)


# ═════════════════════════════════════════════════════════════════════
# Ring buffer
# ═════════════════════════════════════════════════════════════════════

class RingBuffer:
    """Thread-safe ring buffer for continuous EEG data."""

    def __init__(self, max_samples=BUFFER_SAMPLES, n_channels=N_CHANNELS):
        self.max_samples = max_samples
        self.n_channels = n_channels
        self.data = np.zeros((max_samples, n_channels), dtype=np.float64)
        self.timestamps = np.zeros(max_samples, dtype=np.float64)
        self.write_idx = 0
        self.lock = threading.Lock()

    def write(self, samples, timestamps):
        n_new = len(timestamps)
        if n_new == 0:
            return

        with self.lock:
            start_pos = self.write_idx % self.max_samples

            if start_pos + n_new <= self.max_samples:
                self.data[start_pos:start_pos + n_new] = samples
                self.timestamps[start_pos:start_pos + n_new] = timestamps
            else:
                first = self.max_samples - start_pos
                self.data[start_pos:] = samples[:first]
                self.timestamps[start_pos:] = timestamps[:first]
                remainder = n_new - first
                self.data[:remainder] = samples[first:]
                self.timestamps[:remainder] = timestamps[first:]

            self.write_idx += n_new

    def _linearize_locked(self):
        n_valid = min(self.write_idx, self.max_samples)
        if n_valid == 0:
            return (
                np.empty((0, self.n_channels), dtype=self.data.dtype),
                np.empty(0, dtype=self.timestamps.dtype),
            )

        if n_valid == self.max_samples:
            oldest_pos = self.write_idx % self.max_samples
            data_linear = np.concatenate([
                self.data[oldest_pos:],
                self.data[:oldest_pos],
            ], axis=0)
            ts_linear = np.concatenate([
                self.timestamps[oldest_pos:],
                self.timestamps[:oldest_pos],
            ])
        else:
            data_linear = self.data[:n_valid].copy()
            ts_linear = self.timestamps[:n_valid].copy()

        return data_linear, ts_linear

    def snapshot(self):
        """Return the valid buffer contents ordered from oldest to newest."""
        with self.lock:
            return self._linearize_locked()

    def extract_epoch(self, event_timestamp, pre_ms=EPOCH_PRE_MS, post_ms=EPOCH_POST_MS):
        pre_samples = int(pre_ms * SR / 1000)
        post_samples = int(post_ms * SR / 1000)
        epoch_len = pre_samples + post_samples

        with self.lock:
            data_linear, ts_linear = self._linearize_locked()

        if len(ts_linear) < epoch_len:
            return None

        best_idx = _nearest_sample_index(ts_linear, event_timestamp)
        if best_idx is None:
            return None

        epoch_start = best_idx - pre_samples
        epoch_end = best_idx + post_samples

        if epoch_start < 0 or epoch_end > len(ts_linear):
            return None

        return data_linear[epoch_start:epoch_end].copy()


# ═════════════════════════════════════════════════════════════════════
# LSL Receiver
# ═════════════════════════════════════════════════════════════════════

class LSLReceiver(threading.Thread):
    """Background thread that pulls EEG from an LSL stream."""

    def __init__(self, ring_buffer, stream_type="EEG", stream_name=None):
        super().__init__(daemon=True)
        self.ring_buffer = ring_buffer
        self.stream_type = stream_type
        self.stream_name = stream_name
        self.running = False
        self.connected = False
        self.inlet = None
        self._samples_received = 0

    def run(self):
        self.running = True

        print(f"[LSLReceiver] Searching for LSL stream (type='{self.stream_type}')...")
        if self.stream_name:
            streams = resolve_byprop("name", self.stream_name, timeout=30.0)
        else:
            streams = resolve_byprop("type", self.stream_type, timeout=30.0)

        if not streams:
            print("[LSLReceiver] ERROR: No LSL stream found within 30 seconds.")
            self.running = False
            return

        stream_info = streams[0]
        print(f"[LSLReceiver] Found stream: {stream_info.name()} "
              f"({stream_info.channel_count()} ch @ {stream_info.nominal_srate()} Hz)")

        self.inlet = StreamInlet(stream_info, max_buflen=BUFFER_DURATION_S)
        self.connected = True
        print("[LSLReceiver] Connected. Receiving data...")

        while self.running:
            samples, timestamps = self.inlet.pull_chunk(timeout=0.05, max_samples=256)

            if timestamps:
                samples_np = np.array(samples, dtype=np.float64)
                timestamps_np = np.array(timestamps, dtype=np.float64)

                if samples_np.shape[1] > N_CHANNELS:
                    samples_np = samples_np[:, :N_CHANNELS]

                self.ring_buffer.write(samples_np, timestamps_np)
                self._samples_received += len(timestamps)

    def stop(self):
        self.running = False

    @property
    def samples_received(self):
        return self._samples_received


# ═════════════════════════════════════════════════════════════════════
# Main classifier
# ═════════════════════════════════════════════════════════════════════

class RealtimeClassifier:
    """
    Real-time P300 classifier.

    Loads one trained model and its feature configuration, then classifies
    each trial from the continuous EEG buffer.

        classifier = RealtimeClassifier(model_path=...)
        classifier.start()
        classifier.record_flash(direction, timestamp)
        result = classifier.classify_trial()
        classifier.stop()
    """

    def __init__(self, model_path, stream_type="EEG", stream_name=None):
        """
        Args:
            model_path: Path to the model .joblib artifact.
            stream_type: LSL stream type.
            stream_name: Optional LSL stream name.
        """
        self._load_model(model_path)

        # Band-pass filter applied to the continuous buffer (0.5-30 Hz)
        self.b_30hz, self.a_30hz = butter(
            BPF_ORDER, [BPF_LOW, BPF_HIGH], btype="bandpass", fs=SR
        )

        # Ring buffer and LSL receiver
        self.ring_buffer = RingBuffer()
        self.receiver = LSLReceiver(
            self.ring_buffer,
            stream_type=stream_type,
            stream_name=stream_name,
        )

        # Flash events for the current trial
        self.flash_events = []
        self._lock = threading.Lock()

    # ── Loading ──────────────────────────────────────────────────

    def _load_model(self, model_path):
        """Load the model artifact and its feature configuration."""
        print(f"[Classifier] Loading model from {model_path}...")
        artifact = joblib.load(model_path)

        self.model = artifact["model"]
        self.model_name = artifact.get("model_name", "model")

        # ── Parse feature params (handle both saved formats) ─────
        fp = artifact.get("feature_params", {})

        # Feature window (ms post-stimulus)
        if "window_start_ms" in fp:
            self.window_start_ms = int(fp["window_start_ms"])
            self.window_end_ms = int(fp["window_end_ms"])
        elif "window" in fp:
            parts = fp["window"].split("-")
            self.window_start_ms = int(parts[0])
            self.window_end_ms = int(parts[1])
        else:
            self.window_start_ms = 0
            self.window_end_ms = 800

        # Decimation (overlapping moving average)
        if "dec_window" in fp:
            self.dec_window = int(fp["dec_window"])
            self.dec_step = int(fp["dec_step"])
        elif "dec" in fp:
            parts = fp["dec"].split("/")
            self.dec_window = int(parts[0])
            self.dec_step = int(parts[1])
        else:
            self.dec_window = 20
            self.dec_step = 10

        # Channels (None = all 16)
        if "channel_indices" in fp:
            self.channel_indices = list(fp["channel_indices"])
        elif fp.get("channels") == "12ch_no_artifact":
            self.channel_indices = CH_NO_ARTIFACT
        else:
            self.channel_indices = None

        n_ch = len(self.channel_indices) if self.channel_indices else N_CHANNELS
        print(f"[Classifier] Loaded: {self.model_name}")
        print(f"[Classifier]   Window: {self.window_start_ms}-{self.window_end_ms} ms")
        print(f"[Classifier]   Decimation: {self.dec_window}/{self.dec_step}")
        print(f"[Classifier]   Channels: {n_ch}")
        cv_acc = artifact.get("cv_metrics", {}).get("trial_acc", None)
        if cv_acc:
            print(f"[Classifier]   CV trial accuracy: {cv_acc*100:.1f}%")

    # ── Start / Stop / State ─────────────────────────────────────

    def start(self):
        self.receiver.start()
        for _ in range(100):
            if self.receiver.connected:
                print("[Classifier] Ready for real-time classification.")
                return True
            time.sleep(0.05)
        print("[Classifier] WARNING: LSL receiver not connected after 5s.")
        return False

    def stop(self):
        self.receiver.stop()
        print(f"[Classifier] Stopped. Total samples received: "
              f"{self.receiver.samples_received}")

    @property
    def is_connected(self):
        return self.receiver.connected

    # ── Flash event recording ────────────────────────────────────

    def record_flash(self, direction, timestamp=None):
        if timestamp is None:
            timestamp = local_clock()
        with self._lock:
            self.flash_events.append({
                "timestamp": timestamp,
                "direction": direction.lower(),
            })

    def clear_events(self):
        with self._lock:
            self.flash_events.clear()

    def get_event_count(self):
        with self._lock:
            return len(self.flash_events)

    # ── Shared preprocessing ─────────────────────────────────────

    @staticmethod
    def _detect_bad_channels(eeg_filtered):
        channel_stds = eeg_filtered.std(axis=0)
        median_std = np.median(channel_stds)

        bad_channels = []
        for ch in range(eeg_filtered.shape[1]):
            if channel_stds[ch] < BAD_CH_LOW_FACTOR * median_std:
                bad_channels.append(ch)
            elif channel_stds[ch] > BAD_CH_HIGH_FACTOR * median_std:
                bad_channels.append(ch)

        return bad_channels

    @staticmethod
    def _apply_car(eeg, bad_channels):
        good_channels = [ch for ch in range(eeg.shape[1]) if ch not in bad_channels]
        avg = eeg[:, good_channels].mean(axis=1, keepdims=True)
        return eeg - avg

    @staticmethod
    def _apply_baseline_correction(epochs):
        pre_samp = int(EPOCH_PRE_MS * SR / 1000)
        bl_start = pre_samp + int(BASELINE_START_MS * SR / 1000)
        bl_end = pre_samp + int(BASELINE_END_MS * SR / 1000)
        baseline_mean = epochs[:, bl_start:bl_end, :].mean(axis=1, keepdims=True)
        return epochs - baseline_mean

    @staticmethod
    def _artifact_mask(epochs):
        is_clean = np.ones(len(epochs), dtype=bool)
        for i in range(len(epochs)):
            pp = epochs[i].max(axis=0) - epochs[i].min(axis=0)
            if pp.max() > ARTIFACT_PP_THRESHOLD:
                is_clean[i] = False
        return is_clean

    def _preprocess_trial(self, data_linear, ts_linear, events):
        """
        Mirror the offline pipeline on a continuous buffer snapshot.

        Order:
            1. Bandpass the continuous buffer
            2. Detect bad channels on the continuous signal
            3. Apply CAR on the continuous signal
            4. Extract per-flash epochs
            5. Baseline-correct epochs
            6. Reject high peak-to-peak artifacts
        """
        pre_samples = int(EPOCH_PRE_MS * SR / 1000)
        post_samples = int(EPOCH_POST_MS * SR / 1000)

        event_sample_indices = []
        directions = []
        dropped_alignment = 0
        dropped_bounds = 0

        for event in events:
            sample_idx = _nearest_sample_index(ts_linear, event["timestamp"])
            if sample_idx is None:
                dropped_alignment += 1
                continue

            start = sample_idx - pre_samples
            end = sample_idx + post_samples
            if start < 0 or end > len(ts_linear):
                dropped_bounds += 1
                continue

            event_sample_indices.append(sample_idx)
            directions.append(DIRECTION_MAP[event["direction"]])

        if dropped_alignment or dropped_bounds:
            print(
                "[Classifier] Dropped "
                f"{dropped_alignment + dropped_bounds}/{len(events)} flash events "
                f"({dropped_alignment} misaligned, {dropped_bounds} out of bounds)."
            )

        if not event_sample_indices:
            return None, None, None

        eeg_filtered = filtfilt(self.b_30hz, self.a_30hz, data_linear, axis=0)
        bad_channels = self._detect_bad_channels(eeg_filtered)
        if bad_channels:
            print(f"[Classifier] Bad channels detected: {bad_channels}")
        eeg_car = self._apply_car(eeg_filtered, bad_channels)

        raw_epochs = np.array([
            eeg_car[idx - pre_samples:idx + post_samples]
            for idx in event_sample_indices
        ], dtype=np.float64)

        epochs_corrected = self._apply_baseline_correction(raw_epochs)
        is_clean = self._artifact_mask(epochs_corrected)

        n_total = len(raw_epochs)
        n_clean = int(is_clean.sum())
        n_rejected = n_total - n_clean
        if n_rejected > 0:
            print(
                f"[Classifier] Artifact rejection: {n_rejected}/{n_total} "
                f"epochs rejected, {n_clean} clean."
            )

        return epochs_corrected, is_clean, np.array(directions, dtype=np.int8)

    # ── Feature extraction ───────────────────────────────────────

    def _extract_features(self, epochs_corrected, is_clean, directions):
        """
        Extract the feature matrix for the clean epochs.

        Returns:
            X: feature matrix (n_clean, n_features)
            clean_dirs: direction index per clean epoch
        """
        pre_samp = int(EPOCH_PRE_MS * SR / 1000)
        start_idx = pre_samp + int(self.window_start_ms * SR / 1000)
        end_idx = pre_samp + int(self.window_end_ms * SR / 1000)

        if self.channel_indices is None:
            ch_idx = list(range(N_CHANNELS))
        else:
            ch_idx = list(self.channel_indices)
        n_ch = len(ch_idx)

        # The continuous buffer is already band-pass filtered (0.5-30 Hz),
        # so epochs are used as-is here.
        features = []
        clean_dirs = []

        for i in range(len(epochs_corrected)):
            if not is_clean[i]:
                continue

            epoch = epochs_corrected[i][:, ch_idx]     # select channels
            trimmed = epoch[start_idx:end_idx, :]       # time window

            # Overlapping moving average (decimation)
            n_time = trimmed.shape[0]
            n_out = (n_time - self.dec_window) // self.dec_step + 1
            decimated = np.zeros((n_out, n_ch), dtype=np.float64)
            for j in range(n_out):
                s = j * self.dec_step
                decimated[j] = trimmed[s:s + self.dec_window].mean(axis=0)

            features.append(decimated.flatten())
            clean_dirs.append(directions[i])

        X = np.array(features, dtype=np.float64)
        clean_dirs = np.array(clean_dirs, dtype=np.int8)
        return X, clean_dirs

    def _predict_scores(self, X):
        """Run inference and return a 1-D score array."""
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        else:
            # predict_proba → take target-class (column 1) probability
            return self.model.predict_proba(X)[:, 1]

    # ── Classification ───────────────────────────────────────────

    def classify_trial(self, wait_after_last_flash_s=1.0):
        """
        Classify the current trial.

        Returns:
            dict with "direction", "direction_scores", "confidence",
            "n_epochs_used", "n_epochs_total"
            or None if classification fails.
        """
        with self._lock:
            events = list(self.flash_events)

        if not events:
            print("[Classifier] No flash events recorded.")
            return None

        # Wait for the post-stimulus EEG to arrive in the buffer
        last_flash_time = max(e["timestamp"] for e in events)
        min_wait_s = (EPOCH_POST_MS / 1000.0) + 0.1
        buffer_ready_time = last_flash_time + max(wait_after_last_flash_s, min_wait_s)
        now = local_clock()
        if now < buffer_ready_time:
            wait_needed = buffer_ready_time - now
            print(f"[Classifier] Waiting {wait_needed:.2f}s for buffer to fill...")
            time.sleep(wait_needed)

        # ── Step 1: Snapshot the continuous buffer ────────────────
        data_linear, ts_linear = self.ring_buffer.snapshot()
        if len(ts_linear) == 0:
            print("[Classifier] EEG buffer is empty.")
            return None

        # ── Step 2: Offline-matched continuous preprocessing ─────
        epochs_corrected, is_clean, directions = self._preprocess_trial(
            data_linear, ts_linear, events
        )
        if epochs_corrected is None:
            print("[Classifier] Failed to align any flash events to buffered EEG.")
            return None

        n_total = len(epochs_corrected)
        print(f"[Classifier] Extracted {n_total} epochs from continuous buffer.")
        n_clean = int(is_clean.sum())

        if n_clean == 0:
            print("[Classifier] All epochs rejected — cannot classify.")
            return None

        # ── Step 3: Score each clean epoch ───────────────────────
        X, clean_dirs = self._extract_features(
            epochs_corrected, is_clean, directions
        )
        scores = self._predict_scores(X)

        # ── Step 4: Aggregate per direction ──────────────────────
        direction_scores = {}
        for d in range(4):
            d_mask = clean_dirs == d
            if d_mask.sum() > 0:
                direction_scores[d] = float(scores[d_mask].mean())
            else:
                direction_scores[d] = float("-inf")

        predicted_dir = max(direction_scores, key=direction_scores.get)
        predicted_name = DIRECTION_NAMES[predicted_dir]

        sorted_scores = sorted(direction_scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Pretty-print
        print("[Classifier] Scores: " + "  ".join(
            f"{DIRECTION_NAMES[d]}={direction_scores[d]:+.3f}"
            for d in range(4)
        ))
        print(f"[Classifier] → Predicted: {predicted_name.upper()} "
              f"(confidence: {confidence:.3f})")

        return {
            "direction": predicted_name,
            "direction_scores": {
                DIRECTION_NAMES[d]: direction_scores[d] for d in range(4)
            },
            "confidence": confidence,
            "n_epochs_used": n_clean,
            "n_epochs_total": n_total,
        }


# ═════════════════════════════════════════════════════════════════════
# Factory function — creates the classifier from config
# ═════════════════════════════════════════════════════════════════════

def create_classifier(stream_type="EEG", stream_name=None):
    """
    Create a RealtimeClassifier using the settings in config.py.

    Loads the single model at MODEL_PATH.
    """
    from config import MODEL_PATH

    return RealtimeClassifier(
        model_path=MODEL_PATH,
        stream_type=stream_type,
        stream_name=stream_name,
    )


# ═════════════════════════════════════════════════════════════════════
# Standalone test
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    clf = create_classifier()

    if not clf.start():
        print("Could not connect to LSL stream.")
        sys.exit(1)

    try:
        print("\nReceiving EEG data. Press Ctrl+C to stop.\n")
        while True:
            time.sleep(2.0)
            n = clf.receiver.samples_received
            dur = n / SR if n > 0 else 0
            print(f"  Samples: {n} ({dur:.1f}s of data)")
    except KeyboardInterrupt:
        pass

    clf.stop()