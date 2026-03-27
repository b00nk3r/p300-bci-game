"""
Real-Time P300 Classifier for the BCI Game

This module bridges the EEG hardware (via LSL) and the Pygame game.
It receives live EEG data, preprocesses it using the same pipeline
as offline training, and classifies which direction the user attended.

Architecture:
    - A background thread continuously pulls EEG samples from an LSL
      stream and stores them in a thread-safe ring buffer.
    - When the game finishes flashing (trial complete), the main thread
      calls classify_trial() with the list of flash events.
    - classify_trial() extracts epochs from the buffer, preprocesses
      them identically to the offline pipeline, runs the trained LDA,
      and returns the predicted direction.

Usage in the game:
    1. At startup:
        classifier = RealtimeClassifier(model_path="models/single_trial_lda_best_model.joblib")
        classifier.start()

    2. During each flash, record the event:
        classifier.record_flash(direction="up", timestamp=pylsl.local_clock())

    3. When all sequences finish (PROCESSING state):
        predicted_direction = classifier.classify_trial()

    4. Feed result to the game:
        arrow_manager.simulate_selection(predicted_direction)

    5. At shutdown:
        classifier.stop()

Dependencies:
    pip install pylsl joblib scikit-learn scipy numpy
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

SR = 500                  # Sampling rate (Hz) — must match g.Nautilus config
N_CHANNELS = 16           # Number of EEG channels

# Bandpass filter (same as offline config.py)
BPF_LOW = 0.5
BPF_HIGH = 30.0
BPF_ORDER = 4

# Epoch window (same as offline)
EPOCH_PRE_MS = 200
EPOCH_POST_MS = 800
EPOCH_SAMPLES = int((EPOCH_PRE_MS + EPOCH_POST_MS) * SR / 1000)  # 500 samples

# Baseline correction window
BASELINE_START_MS = -100
BASELINE_END_MS = 0

# Artifact rejection
ARTIFACT_PP_THRESHOLD = 150.0  # µV peak-to-peak

# Bad channel detection
BAD_CH_LOW_FACTOR = 0.01
BAD_CH_HIGH_FACTOR = 4.0

# Ring buffer: 60 seconds of data (generous margin)
BUFFER_DURATION_S = 60
BUFFER_SAMPLES = BUFFER_DURATION_S * SR  # 30000 samples

# Direction mapping (same as offline config.py)
DIRECTION_MAP = {"up": 0, "down": 1, "left": 2, "right": 3}
DIRECTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


class RingBuffer:
    """
    Thread-safe ring buffer for continuous EEG data.

    Stores (samples, channels) EEG data plus an LSL timestamp per sample.
    The receiver thread writes; the main thread reads for epoch extraction.
    """

    def __init__(self, max_samples=BUFFER_SAMPLES, n_channels=N_CHANNELS):
        self.max_samples = max_samples
        self.n_channels = n_channels
        self.data = np.zeros((max_samples, n_channels), dtype=np.float64)
        self.timestamps = np.zeros(max_samples, dtype=np.float64)
        self.write_idx = 0        # Total samples written (monotonic)
        self.lock = threading.Lock()

    def write(self, samples, timestamps):
        """
        Write a chunk of samples into the buffer.

        Args:
            samples: np.ndarray of shape (n_new, n_channels)
            timestamps: np.ndarray of shape (n_new,) — LSL timestamps
        """
        n_new = len(timestamps)
        if n_new == 0:
            return

        with self.lock:
            start_pos = self.write_idx % self.max_samples

            if start_pos + n_new <= self.max_samples:
                # No wrap-around
                self.data[start_pos:start_pos + n_new] = samples
                self.timestamps[start_pos:start_pos + n_new] = timestamps
            else:
                # Wrap-around: write in two parts
                first = self.max_samples - start_pos
                self.data[start_pos:] = samples[:first]
                self.timestamps[start_pos:] = timestamps[:first]
                remainder = n_new - first
                self.data[:remainder] = samples[first:]
                self.timestamps[:remainder] = timestamps[first:]

            self.write_idx += n_new

    def extract_epoch(self, event_timestamp, pre_ms=EPOCH_PRE_MS, post_ms=EPOCH_POST_MS):
        """
        Extract an epoch centered on an event timestamp.

        Args:
            event_timestamp: LSL timestamp of the flash event
            pre_ms: milliseconds before the event (positive)
            post_ms: milliseconds after the event (positive)

        Returns:
            np.ndarray of shape (epoch_samples, n_channels), or None if
            the required data isn't available in the buffer.
        """
        pre_samples = int(pre_ms * SR / 1000)
        post_samples = int(post_ms * SR / 1000)
        epoch_len = pre_samples + post_samples

        with self.lock:
            if self.write_idx < epoch_len:
                return None  # Not enough data yet

            # Find the buffer index closest to event_timestamp.
            # We search only the valid portion of the buffer.
            n_valid = min(self.write_idx, self.max_samples)
            if n_valid == self.max_samples:
                # Buffer is full; valid region is the whole array,
                # but we need to handle the circular ordering.
                oldest_pos = self.write_idx % self.max_samples
                # Reconstruct linearized timestamps for searching
                ts_linear = np.concatenate([
                    self.timestamps[oldest_pos:],
                    self.timestamps[:oldest_pos],
                ])
            else:
                ts_linear = self.timestamps[:n_valid]

            # Binary search for the closest timestamp
            insert_idx = np.searchsorted(ts_linear, event_timestamp)
            # Check neighbors to find the actual closest
            best_idx = insert_idx
            if insert_idx > 0:
                if insert_idx >= len(ts_linear) or \
                   abs(ts_linear[insert_idx - 1] - event_timestamp) < \
                   abs(ts_linear[min(insert_idx, len(ts_linear) - 1)] - event_timestamp):
                    best_idx = insert_idx - 1

            if best_idx >= len(ts_linear):
                best_idx = len(ts_linear) - 1

            # Check that the match is reasonably close (within 10ms)
            if abs(ts_linear[best_idx] - event_timestamp) > 0.010:
                return None  # Timestamp too far from any sample

            # Extract the epoch from the linearized view
            epoch_start = best_idx - pre_samples
            epoch_end = best_idx + post_samples

            if epoch_start < 0 or epoch_end > len(ts_linear):
                return None  # Epoch extends outside available data

            # Map back to the actual buffer positions
            if n_valid == self.max_samples:
                data_linear = np.concatenate([
                    self.data[oldest_pos:],
                    self.data[:oldest_pos],
                ], axis=0)
            else:
                data_linear = self.data[:n_valid]

            epoch = data_linear[epoch_start:epoch_end].copy()
            return epoch


class LSLReceiver(threading.Thread):
    """
    Background thread that continuously receives EEG from an LSL stream.

    Resolves the stream on start, then pulls chunks in a loop and writes
    them into the ring buffer. Runs as a daemon thread so it doesn't
    block program exit.
    """

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

        # Resolve the LSL stream
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
            # Pull a chunk of samples (non-blocking with short timeout)
            samples, timestamps = self.inlet.pull_chunk(timeout=0.05, max_samples=256)

            if timestamps:
                samples_np = np.array(samples, dtype=np.float64)
                timestamps_np = np.array(timestamps, dtype=np.float64)

                # If the stream has more channels than we need, take first N_CHANNELS
                if samples_np.shape[1] > N_CHANNELS:
                    samples_np = samples_np[:, :N_CHANNELS]

                self.ring_buffer.write(samples_np, timestamps_np)
                self._samples_received += len(timestamps)

    def stop(self):
        self.running = False

    @property
    def samples_received(self):
        return self._samples_received


class RealtimeClassifier:
    """
    Main integration class. Manages the LSL receiver, stores flash events,
    and runs the trained LDA classifier on extracted epochs.

    This class is what the game interacts with directly.
    """

    def __init__(self, model_path, stream_type="EEG", stream_name=None):
        """
        Args:
            model_path: Path to the saved joblib artifact from training.
                        Contains the LDA model and feature extraction params.
            stream_type: LSL stream type to resolve (default "EEG").
            stream_name: Optional specific LSL stream name.
        """
        # Load the trained model artifact
        print(f"[Classifier] Loading model from {model_path}...")
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_params = artifact["feature_params"]

        print(f"[Classifier] Model loaded: {artifact['model_name']}")
        print(f"[Classifier]   Window: {self.feature_params['window_start_ms']}"
              f"-{self.feature_params['window_end_ms']} ms")
        print(f"[Classifier]   Decimation: {self.feature_params['dec_window']}"
              f"/{self.feature_params['dec_step']}")
        print(f"[Classifier]   CV trial accuracy: "
              f"{artifact['cv_metrics']['trial_acc']*100:.1f}%")

        # Pre-compute the bandpass filter coefficients (same as offline)
        self.b, self.a = butter(
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
        # Each entry: {"timestamp": float, "direction": str}
        self.flash_events = []
        self._lock = threading.Lock()

    def start(self):
        """Start the LSL receiver thread."""
        self.receiver.start()
        # Wait briefly for connection
        for _ in range(100):  # Up to 5 seconds
            if self.receiver.connected:
                print("[Classifier] Ready for real-time classification.")
                return True
            time.sleep(0.05)
        print("[Classifier] WARNING: LSL receiver not connected after 5s.")
        return False

    def stop(self):
        """Stop the LSL receiver thread."""
        self.receiver.stop()
        print(f"[Classifier] Stopped. Total samples received: "
              f"{self.receiver.samples_received}")

    @property
    def is_connected(self):
        return self.receiver.connected

    # ── Flash event recording ────────────────────────────────────────

    def record_flash(self, direction, timestamp=None):
        """
        Record a flash event. Call this every time an arrow flashes.

        Args:
            direction: str — "up", "down", "left", or "right"
            timestamp: LSL timestamp (from pylsl.local_clock()).
                       If None, uses current time.
        """
        if timestamp is None:
            timestamp = local_clock()

        with self._lock:
            self.flash_events.append({
                "timestamp": timestamp,
                "direction": direction.lower(),
            })

    def clear_events(self):
        """Clear recorded flash events (call at start of new trial)."""
        with self._lock:
            self.flash_events.clear()

    def get_event_count(self):
        """Number of flash events recorded for the current trial."""
        with self._lock:
            return len(self.flash_events)

    # ── Classification ───────────────────────────────────────────────

    def classify_trial(self, wait_after_last_flash_s=1.0):
        """
        Classify the current trial's flash events.

        Extracts epochs from the ring buffer, preprocesses them using
        the same pipeline as offline training, runs the LDA on each
        epoch, and returns the direction with the highest average score.

        Args:
            wait_after_last_flash_s: seconds to wait after the last flash
                before extracting epochs (ensures the post-stimulus window
                is fully captured in the buffer).

        Returns:
            dict with:
                "direction": str — predicted direction ("up"/"down"/"left"/"right")
                "direction_scores": dict — {direction_name: mean_score}
                "confidence": float — margin between top and second score
                "n_epochs_used": int — clean epochs after artifact rejection
                "n_epochs_total": int — total epochs before rejection
            or None if classification fails.
        """
        with self._lock:
            events = list(self.flash_events)

        if not events:
            print("[Classifier] No flash events recorded.")
            return None

        # Wait for the post-stimulus EEG to arrive in the buffer
        last_flash_time = max(e["timestamp"] for e in events)
        buffer_ready_time = last_flash_time + (EPOCH_POST_MS / 1000.0) + 0.1
        now = local_clock()
        if now < buffer_ready_time:
            wait_needed = buffer_ready_time - now
            print(f"[Classifier] Waiting {wait_needed:.2f}s for buffer to fill...")
            time.sleep(wait_needed)

        # ── Step 1: Extract epochs from ring buffer ──────────────────

        raw_epochs = []
        epoch_directions = []

        for event in events:
            epoch = self.ring_buffer.extract_epoch(event["timestamp"])
            if epoch is not None:
                raw_epochs.append(epoch)
                epoch_directions.append(DIRECTION_MAP[event["direction"]])

        if not raw_epochs:
            print("[Classifier] Failed to extract any epochs from buffer.")
            return None

        # Stack into array: (n_epochs, epoch_samples, n_channels)
        epochs_raw = np.array(raw_epochs, dtype=np.float64)
        directions = np.array(epoch_directions, dtype=np.int8)
        n_total = len(epochs_raw)

        print(f"[Classifier] Extracted {n_total} epochs from buffer.")

        # ── Step 2: Bandpass filter each epoch ───────────────────────
        # Apply per-epoch to avoid edge effects between epochs
        epochs_filtered = np.zeros_like(epochs_raw)
        for i in range(n_total):
            epochs_filtered[i] = filtfilt(self.b, self.a, epochs_raw[i], axis=0)

        # ── Step 3: Detect bad channels and apply CAR ────────────────
        # Use the concatenated filtered data for channel statistics
        all_data = epochs_filtered.reshape(-1, N_CHANNELS)
        channel_stds = all_data.std(axis=0)
        median_std = np.median(channel_stds)

        bad_channels = []
        for ch in range(N_CHANNELS):
            if channel_stds[ch] < BAD_CH_LOW_FACTOR * median_std:
                bad_channels.append(ch)
            elif channel_stds[ch] > BAD_CH_HIGH_FACTOR * median_std:
                bad_channels.append(ch)

        good_channels = [ch for ch in range(N_CHANNELS) if ch not in bad_channels]

        if bad_channels:
            print(f"[Classifier] Bad channels detected: {bad_channels}")

        # Apply CAR (Common Average Reference) excluding bad channels
        for i in range(n_total):
            avg = epochs_filtered[i][:, good_channels].mean(axis=1, keepdims=True)
            epochs_filtered[i] = epochs_filtered[i] - avg

        # ── Step 4: Baseline correction ──────────────────────────────
        pre_samp = int(EPOCH_PRE_MS * SR / 1000)
        bl_start = pre_samp + int(BASELINE_START_MS * SR / 1000)
        bl_end = pre_samp + int(BASELINE_END_MS * SR / 1000)

        baseline_mean = epochs_filtered[:, bl_start:bl_end, :].mean(
            axis=1, keepdims=True
        )
        epochs_corrected = epochs_filtered - baseline_mean

        # ── Step 5: Artifact rejection ───────────────────────────────
        is_clean = np.ones(n_total, dtype=bool)
        for i in range(n_total):
            pp = epochs_corrected[i].max(axis=0) - epochs_corrected[i].min(axis=0)
            if pp.max() > ARTIFACT_PP_THRESHOLD:
                is_clean[i] = False

        n_clean = is_clean.sum()
        n_rejected = n_total - n_clean
        if n_rejected > 0:
            print(f"[Classifier] Artifact rejection: {n_rejected}/{n_total} "
                  f"epochs rejected, {n_clean} clean.")

        if n_clean == 0:
            print("[Classifier] All epochs rejected — cannot classify.")
            return None

        # ── Step 6: Feature extraction ───────────────────────────────
        w_start = self.feature_params["window_start_ms"]
        w_end = self.feature_params["window_end_ms"]
        dec_window = self.feature_params["dec_window"]
        dec_step = self.feature_params["dec_step"]

        start_idx = pre_samp + int(w_start * SR / 1000)
        end_idx = pre_samp + int(w_end * SR / 1000)

        features = []
        clean_directions = []

        for i in range(n_total):
            if not is_clean[i]:
                continue

            epoch = epochs_corrected[i]
            trimmed = epoch[start_idx:end_idx, :]

            # Overlapping moving average (decimation) — same as offline
            n_time = trimmed.shape[0]
            n_out = (n_time - dec_window) // dec_step + 1
            decimated = np.zeros((n_out, N_CHANNELS), dtype=np.float64)
            for j in range(n_out):
                s = j * dec_step
                decimated[j] = trimmed[s:s + dec_window].mean(axis=0)

            features.append(decimated.flatten())
            clean_directions.append(directions[i])

        X = np.array(features, dtype=np.float64)
        clean_dirs = np.array(clean_directions, dtype=np.int8)

        # ── Step 7: Run LDA classifier ───────────────────────────────
        scores = self.model.decision_function(X)

        # ── Step 8: Aggregate scores per direction ───────────────────
        direction_scores = {}
        for d in range(4):
            d_mask = clean_dirs == d
            if d_mask.sum() > 0:
                direction_scores[d] = float(scores[d_mask].mean())
            else:
                direction_scores[d] = float("-inf")

        predicted_dir = max(direction_scores, key=direction_scores.get)
        predicted_name = DIRECTION_NAMES[predicted_dir]

        # Confidence = margin between top two scores
        sorted_scores = sorted(direction_scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Pretty-print results
        print(f"[Classifier] Scores: " + "  ".join(
            f"{DIRECTION_NAMES[d]}={direction_scores[d]:+.3f}"
            for d in range(4)
        ))
        print(f"[Classifier] → Predicted: {predicted_name.upper()} "
              f"(confidence: {confidence:.3f})")

        return {
            "direction": predicted_name,
            "direction_scores": {DIRECTION_NAMES[d]: direction_scores[d] for d in range(4)},
            "confidence": confidence,
            "n_epochs_used": int(n_clean),
            "n_epochs_total": n_total,
        }


# ── Standalone test ──────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test: connect to an LSL stream and print buffer stats.
    Does NOT run classification (no flash events to classify).
    """
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/single_trial_lda_best_model.joblib"

    clf = RealtimeClassifier(model_path=model_path)

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
