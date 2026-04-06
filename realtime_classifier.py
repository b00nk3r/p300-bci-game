"""
Real-Time P300 Classifier for the BCI Game
===========================================

Supports two modes:
    - Single model: loads one .joblib artifact (LDA, LR, etc.)
    - Ensemble: loads multiple models with different feature configs,
      z-score normalises their scores, and combines with learned weights.

The mode is controlled by MODEL_MODE in config.py ("single" or "ensemble").

Architecture:
    - A background thread continuously pulls EEG samples from an LSL
      stream and stores them in a thread-safe ring buffer.
    - When the game finishes flashing (trial complete), the main thread
      calls classify_trial() with the list of flash events.
    - classify_trial() extracts epochs, preprocesses them, runs the
      model(s), and returns the predicted direction.

Interface (unchanged from single-model version):
    classifier = RealtimeClassifier(...)
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

SR = 500
N_CHANNELS = 16

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

DIRECTION_MAP = {"up": 0, "down": 1, "left": 2, "right": 3}
DIRECTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

# Channels to drop for the "12ch no artifact" subset
# FP1=0, FP2=1, T7=5, T8=9  →  keep the rest
CH_NO_ARTIFACT = [2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15]


# ═════════════════════════════════════════════════════════════════════
# Ring buffer (unchanged)
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

    def extract_epoch(self, event_timestamp, pre_ms=EPOCH_PRE_MS, post_ms=EPOCH_POST_MS):
        pre_samples = int(pre_ms * SR / 1000)
        post_samples = int(post_ms * SR / 1000)
        epoch_len = pre_samples + post_samples

        with self.lock:
            if self.write_idx < epoch_len:
                return None

            n_valid = min(self.write_idx, self.max_samples)
            if n_valid == self.max_samples:
                oldest_pos = self.write_idx % self.max_samples
                ts_linear = np.concatenate([
                    self.timestamps[oldest_pos:],
                    self.timestamps[:oldest_pos],
                ])
            else:
                ts_linear = self.timestamps[:n_valid]

            insert_idx = np.searchsorted(ts_linear, event_timestamp)
            best_idx = insert_idx
            if insert_idx > 0:
                if insert_idx >= len(ts_linear) or \
                   abs(ts_linear[insert_idx - 1] - event_timestamp) < \
                   abs(ts_linear[min(insert_idx, len(ts_linear) - 1)] - event_timestamp):
                    best_idx = insert_idx - 1

            if best_idx >= len(ts_linear):
                best_idx = len(ts_linear) - 1

            if abs(ts_linear[best_idx] - event_timestamp) > 0.010:
                return None

            epoch_start = best_idx - pre_samples
            epoch_end = best_idx + post_samples

            if epoch_start < 0 or epoch_end > len(ts_linear):
                return None

            if n_valid == self.max_samples:
                data_linear = np.concatenate([
                    self.data[oldest_pos:],
                    self.data[:oldest_pos],
                ], axis=0)
            else:
                data_linear = self.data[:n_valid]

            epoch = data_linear[epoch_start:epoch_end].copy()
            return epoch


# ═════════════════════════════════════════════════════════════════════
# LSL Receiver (unchanged)
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
# Model descriptor — stores one model + its feature config
# ═════════════════════════════════════════════════════════════════════

class ModelDescriptor:
    """
    Everything needed to run one model at inference time.

    Attributes:
        name:           Human-readable name (e.g. "1_LDA")
        model:          Fitted sklearn estimator
        weight:         Ensemble weight (sums to ~1 across all models)
        dec_window:     Decimation window size in samples
        dec_step:       Decimation step in samples
        lowpass_hz:     Low-pass cutoff applied to epochs (30 = standard, <30 = extra filter)
        channel_indices: Which channels to use (None = all 16)
        window_start_ms: Feature window start (ms post-stimulus)
        window_end_ms:   Feature window end
        score_mean:     Mean of CV scores (for z-score normalisation)
        score_std:      Std of CV scores
    """

    def __init__(self, name, artifact, weight):
        self.name = name
        self.model = artifact["model"]
        self.weight = weight

        # ── Parse feature params (handle both formats) ───────────
        fp = artifact.get("feature_params", {})

        # Window
        if "window_start_ms" in fp:
            self.window_start_ms = int(fp["window_start_ms"])
            self.window_end_ms = int(fp["window_end_ms"])
        elif "window" in fp:
            parts = fp["window"].split("-")
            self.window_start_ms = int(parts[0])
            self.window_end_ms = int(parts[1])
        else:
            self.window_start_ms = 0
            self.window_end_ms = 600

        # Decimation
        if "dec_window" in fp:
            self.dec_window = int(fp["dec_window"])
            self.dec_step = int(fp["dec_step"])
        elif "dec" in fp:
            parts = fp["dec"].split("/")
            self.dec_window = int(parts[0])
            self.dec_step = int(parts[1])
        else:
            self.dec_window = 30
            self.dec_step = 15

        # Low-pass cutoff
        self.lowpass_hz = float(fp.get("lowpass", 30))

        # Channels
        if "channel_indices" in fp:
            self.channel_indices = list(fp["channel_indices"])
        elif fp.get("channels") == "12ch_no_artifact":
            self.channel_indices = CH_NO_ARTIFACT
        else:
            self.channel_indices = None  # all 16

        # ── Normalisation stats from CV scores ───────────────────
        cv_scores = artifact.get("cv_scores", None)
        if cv_scores is not None:
            self.score_mean = float(np.mean(cv_scores))
            self.score_std = float(np.std(cv_scores))
            if self.score_std < 1e-10:
                self.score_std = 1.0
        else:
            self.score_mean = 0.0
            self.score_std = 1.0

    @property
    def feature_key(self):
        """
        Unique key for the feature extraction pipeline.
        Models sharing the same key can reuse the same feature matrix.
        """
        ch_key = tuple(self.channel_indices) if self.channel_indices else "all"
        return (
            self.window_start_ms, self.window_end_ms,
            self.dec_window, self.dec_step,
            self.lowpass_hz,
            ch_key,
        )

    def predict_scores(self, X):
        """Run inference and return 1-D score array."""
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        else:
            # predict_proba → take target-class (column 1) probability
            return self.model.predict_proba(X)[:, 1]

    def normalise(self, scores):
        """Z-score normalise using training statistics."""
        return (scores - self.score_mean) / self.score_std

    def __repr__(self):
        ch = len(self.channel_indices) if self.channel_indices else 16
        return (f"ModelDescriptor({self.name}, dec={self.dec_window}/{self.dec_step}, "
                f"lp={self.lowpass_hz}Hz, ch={ch}, w={self.weight:.3f})")


# ═════════════════════════════════════════════════════════════════════
# Main classifier
# ═════════════════════════════════════════════════════════════════════

class RealtimeClassifier:
    """
    Real-time P300 classifier supporting single-model and ensemble modes.

    The external interface is identical in both modes:
        classifier.start()
        classifier.record_flash(direction, timestamp)
        result = classifier.classify_trial()
        classifier.stop()
    """

    def __init__(
        self,
        model_path=None,
        ensemble_model_paths=None,
        ensemble_weights=None,
        stream_type="EEG",
        stream_name=None,
    ):
        """
        Args:
            model_path: Path to single model .joblib (single mode).
            ensemble_model_paths: Dict {name: path} for all ensemble models.
            ensemble_weights: Dict {name: weight} for ensemble combining.
            stream_type: LSL stream type.
            stream_name: Optional LSL stream name.
        """
        self.ensemble_mode = (ensemble_model_paths is not None)

        if self.ensemble_mode:
            self._load_ensemble(ensemble_model_paths, ensemble_weights or {})
        else:
            self._load_single(model_path)

        # Bandpass filter coefficients (shared across all models)
        self.b_30hz, self.a_30hz = butter(
            BPF_ORDER, [BPF_LOW, BPF_HIGH], btype="bandpass", fs=SR
        )

        # Extra low-pass filters (pre-computed for any model that needs them)
        self._lowpass_filters = {}
        if self.ensemble_mode:
            for md in self.models:
                if md.lowpass_hz < BPF_HIGH:
                    if md.lowpass_hz not in self._lowpass_filters:
                        b, a = butter(4, md.lowpass_hz, btype="low", fs=SR)
                        self._lowpass_filters[md.lowpass_hz] = (b, a)

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

    def _load_single(self, model_path):
        """Load a single model artifact."""
        print(f"[Classifier] Loading single model from {model_path}...")
        artifact = joblib.load(model_path)

        self.models = [ModelDescriptor(
            name=artifact.get("model_name", "single"),
            artifact=artifact,
            weight=1.0,
        )]
        self.feature_groups = {self.models[0].feature_key: [self.models[0]]}

        fp = self.models[0]
        print(f"[Classifier] Loaded: {fp.name}")
        print(f"[Classifier]   Window: {fp.window_start_ms}-{fp.window_end_ms} ms")
        print(f"[Classifier]   Decimation: {fp.dec_window}/{fp.dec_step}")
        cv_acc = artifact.get("cv_metrics", {}).get("trial_acc", None)
        if cv_acc:
            print(f"[Classifier]   CV trial accuracy: {cv_acc*100:.1f}%")

    def _load_ensemble(self, model_paths, weights):
        """Load all ensemble model artifacts and group by feature config."""
        print(f"[Classifier] Loading ensemble ({len(model_paths)} models)...")

        self.models = []
        for name, path in model_paths.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                print(f"[Classifier]   Skipping {name} (weight=0)")
                continue

            try:
                artifact = joblib.load(path)
            except FileNotFoundError:
                print(f"[Classifier]   WARNING: {path} not found — skipping {name}")
                continue

            md = ModelDescriptor(name, artifact, w)
            self.models.append(md)

            cv_acc = artifact.get("cv_metrics", {}).get("trial_acc", None)
            acc_str = f"  cv={cv_acc*100:.1f}%" if cv_acc else ""
            print(f"[Classifier]   {md}{acc_str}")

        # Renormalise weights so they sum to 1
        total_w = sum(md.weight for md in self.models)
        if total_w > 0:
            for md in self.models:
                md.weight /= total_w

        # Group models by feature pipeline (models sharing the same key
        # will reuse the same feature extraction)
        self.feature_groups = {}
        for md in self.models:
            self.feature_groups.setdefault(md.feature_key, []).append(md)

        print(f"[Classifier] {len(self.models)} models loaded, "
              f"{len(self.feature_groups)} unique feature pipelines")

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

    def _preprocess_epochs(self, raw_epochs):
        """
        Shared preprocessing: bandpass → bad channels → CAR → baseline → artifact rejection.

        Args:
            raw_epochs: np.ndarray of shape (n_epochs, epoch_samples, n_channels)

        Returns:
            epochs_corrected: preprocessed epochs (n_clean, epoch_samples, n_channels)
            is_clean: boolean mask
        """
        n_total = len(raw_epochs)

        # Bandpass filter (0.5-30 Hz)
        epochs_filtered = np.zeros_like(raw_epochs)
        for i in range(n_total):
            epochs_filtered[i] = filtfilt(self.b_30hz, self.a_30hz, raw_epochs[i], axis=0)

        # Detect bad channels
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

        # CAR
        for i in range(n_total):
            avg = epochs_filtered[i][:, good_channels].mean(axis=1, keepdims=True)
            epochs_filtered[i] = epochs_filtered[i] - avg

        # Baseline correction
        pre_samp = int(EPOCH_PRE_MS * SR / 1000)
        bl_start = pre_samp + int(BASELINE_START_MS * SR / 1000)
        bl_end = pre_samp + int(BASELINE_END_MS * SR / 1000)

        baseline_mean = epochs_filtered[:, bl_start:bl_end, :].mean(
            axis=1, keepdims=True
        )
        epochs_corrected = epochs_filtered - baseline_mean

        # Artifact rejection
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

        return epochs_corrected, is_clean

    # ── Feature extraction for one pipeline config ───────────────

    def _extract_features(self, epochs_corrected, is_clean, directions, feature_key):
        """
        Extract features for one unique feature pipeline.

        Args:
            epochs_corrected: (n_total, epoch_samples, n_channels) — full preprocessed
            is_clean: boolean mask
            directions: direction index per epoch
            feature_key: (w_start, w_end, dec_w, dec_s, lowpass, ch_key)

        Returns:
            X: feature matrix (n_clean, n_features)
            clean_dirs: direction indices for clean epochs
        """
        w_start_ms, w_end_ms, dec_window, dec_step, lowpass_hz, ch_key = feature_key

        pre_samp = int(EPOCH_PRE_MS * SR / 1000)
        start_idx = pre_samp + int(w_start_ms * SR / 1000)
        end_idx = pre_samp + int(w_end_ms * SR / 1000)

        # Determine channel indices
        if ch_key == "all":
            ch_idx = list(range(N_CHANNELS))
        else:
            ch_idx = list(ch_key)
        n_ch = len(ch_idx)

        # Apply extra low-pass if needed (e.g. 10 Hz)
        if lowpass_hz < BPF_HIGH and lowpass_hz in self._lowpass_filters:
            b_lp, a_lp = self._lowpass_filters[lowpass_hz]
            epochs_to_use = np.zeros_like(epochs_corrected)
            for i in range(len(epochs_corrected)):
                epochs_to_use[i] = filtfilt(b_lp, a_lp, epochs_corrected[i], axis=0)
        else:
            epochs_to_use = epochs_corrected

        features = []
        clean_dirs = []

        for i in range(len(epochs_to_use)):
            if not is_clean[i]:
                continue

            epoch = epochs_to_use[i][:, ch_idx]       # select channels
            trimmed = epoch[start_idx:end_idx, :]      # time window

            # Overlapping moving average (decimation)
            n_time = trimmed.shape[0]
            n_out = (n_time - dec_window) // dec_step + 1
            decimated = np.zeros((n_out, n_ch), dtype=np.float64)
            for j in range(n_out):
                s = j * dec_step
                decimated[j] = trimmed[s:s + dec_window].mean(axis=0)

            features.append(decimated.flatten())
            clean_dirs.append(directions[i])

        X = np.array(features, dtype=np.float64)
        clean_dirs = np.array(clean_dirs, dtype=np.int8)
        return X, clean_dirs

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
        buffer_ready_time = last_flash_time + (EPOCH_POST_MS / 1000.0) + 0.1
        now = local_clock()
        if now < buffer_ready_time:
            wait_needed = buffer_ready_time - now
            print(f"[Classifier] Waiting {wait_needed:.2f}s for buffer to fill...")
            time.sleep(wait_needed)

        # ── Step 1: Extract epochs from ring buffer ──────────────
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

        epochs_raw = np.array(raw_epochs, dtype=np.float64)
        directions = np.array(epoch_directions, dtype=np.int8)
        n_total = len(epochs_raw)

        print(f"[Classifier] Extracted {n_total} epochs from buffer.")

        # ── Step 2: Shared preprocessing ─────────────────────────
        epochs_corrected, is_clean = self._preprocess_epochs(epochs_raw)
        n_clean = int(is_clean.sum())

        if n_clean == 0:
            print("[Classifier] All epochs rejected — cannot classify.")
            return None

        # ── Step 3: Run each feature group & collect scores ──────
        #
        # For each unique feature pipeline, extract features once,
        # then run all models that share that pipeline.
        # Each model's scores are z-normalised and weighted.

        # Accumulate weighted normalised scores per epoch (clean only)
        combined_scores = np.zeros(n_clean, dtype=np.float64)
        # We also need clean_dirs — same for all groups (same is_clean mask)
        clean_dirs_ref = None

        for feature_key, model_list in self.feature_groups.items():
            X, clean_dirs = self._extract_features(
                epochs_corrected, is_clean, directions, feature_key
            )

            if clean_dirs_ref is None:
                clean_dirs_ref = clean_dirs

            for md in model_list:
                raw_scores = md.predict_scores(X)
                norm_scores = md.normalise(raw_scores)
                combined_scores += norm_scores * md.weight

        # ── Step 4: Aggregate per direction ──────────────────────
        direction_scores = {}
        for d in range(4):
            d_mask = clean_dirs_ref == d
            if d_mask.sum() > 0:
                direction_scores[d] = float(combined_scores[d_mask].mean())
            else:
                direction_scores[d] = float("-inf")

        predicted_dir = max(direction_scores, key=direction_scores.get)
        predicted_name = DIRECTION_NAMES[predicted_dir]

        sorted_scores = sorted(direction_scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Pretty-print
        mode_str = f"ensemble ({len(self.models)} models)" if self.ensemble_mode else "single"
        print(f"[Classifier] [{mode_str}] Scores: " + "  ".join(
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
# Factory function — creates the right classifier based on config
# ═════════════════════════════════════════════════════════════════════

def create_classifier(stream_type="EEG", stream_name=None):
    """
    Create a RealtimeClassifier using the settings in config.py.

    Returns a classifier in single or ensemble mode depending on MODEL_MODE.
    """
    from config import MODEL_MODE, MODEL_PATH

    if MODEL_MODE == "ensemble":
        from config import ENSEMBLE_MODEL_PATHS, ENSEMBLE_WEIGHTS
        return RealtimeClassifier(
            ensemble_model_paths=ENSEMBLE_MODEL_PATHS,
            ensemble_weights=ENSEMBLE_WEIGHTS,
            stream_type=stream_type,
            stream_name=stream_name,
        )
    else:
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