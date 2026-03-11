"""
BCI Controller
==============
High-level coordinator between EEGReceiver, RealtimeClassifier,
and the game.  Provides a simple interface for ArrowManager / main.py:

    controller.begin_trial()
    controller.record_flash(direction_string)
    result = controller.end_trial()
"""

from bci_config import EPOCH_PRE_MS, EPOCH_POST_MS, SR

try:
    import pylsl
    HAS_PYLSL = True
except ImportError:
    HAS_PYLSL = False


class BCIController:
    """Coordinates EEG acquisition, preprocessing, and classification."""

    def __init__(self, model_path: str):
        from src.communication.eeg_receiver import EEGReceiver
        from src.communication.realtime_preprocessor import RealtimeClassifier

        self._receiver = EEGReceiver()
        self._classifier = RealtimeClassifier(model_path)

        self._flash_events = []
        self._trial_start_time = None
        self._bad_channels = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 10.0):
        """Start the EEG receiver (blocks briefly while resolving LSL)."""
        self._receiver.start(timeout=timeout)

    def stop(self):
        """Stop the EEG receiver."""
        self._receiver.stop()

    def is_connected(self) -> bool:
        """True if EEG data is actively being received."""
        return self._receiver.is_connected()

    # ------------------------------------------------------------------
    # Trial flow
    # ------------------------------------------------------------------

    def begin_trial(self):
        """Called when a trial starts (e.g. SPACE pressed).

        Clears flash event list and records the trial start time.
        """
        self._flash_events = []
        if HAS_PYLSL:
            self._trial_start_time = pylsl.local_clock()

    def record_flash(self, direction: str):
        """Record a single flash event during the trial.

        Args:
            direction: One of "up", "down", "left", "right".
        """
        if HAS_PYLSL:
            self._flash_events.append({
                "time": pylsl.local_clock(),
                "direction": direction,
            })

    def end_trial(self):
        """Called when all flashes are done (state → PROCESSING).

        Retrieves the relevant EEG chunk from the buffer, runs the
        full preprocessing + classification pipeline, and returns
        the result dict.

        Returns:
            dict  – see RealtimeClassifier.classify_trial() for keys,
                    or None if there are no flash events / no EEG data.
        """
        if not self._flash_events:
            return None

        flash_times = [ev["time"] for ev in self._flash_events]
        earliest = min(flash_times)
        latest = max(flash_times)

        margin_s = 0.5  # extra data for filter edge effects
        pre_s = EPOCH_PRE_MS / 1000.0
        post_s = EPOCH_POST_MS / 1000.0

        start_time = earliest - pre_s - margin_s
        end_time = latest + post_s + margin_s

        eeg_chunk, timestamps = self._receiver.get_chunk(start_time, end_time)

        if eeg_chunk.shape[0] == 0:
            print("WARNING: No EEG data in buffer for the trial time range")
            return None

        chunk_start_time = timestamps[0]

        result = self._classifier.classify_trial(
            eeg_chunk,
            chunk_start_time,
            self._flash_events,
            bad_channels=self._bad_channels,
        )

        return result

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def run_calibration(self, duration_s: float = 10.0):
        """Collect resting EEG and detect bad channels.

        Call once at session start before gameplay begins.
        """
        if not HAS_PYLSL or not self.is_connected():
            return

        import time
        from src.communication.realtime_preprocessor import (
            apply_bandpass_filter,
            detect_bad_channels,
        )

        print(f"BCI calibration: collecting {duration_s}s of resting EEG...")
        time.sleep(duration_s)

        now = pylsl.local_clock()
        eeg_chunk, _ = self._receiver.get_chunk(now - duration_s, now)

        if eeg_chunk.shape[0] < SR * 2:
            print("WARNING: Not enough EEG data for calibration")
            return

        filtered = apply_bandpass_filter(eeg_chunk)
        self._bad_channels, stds = detect_bad_channels(filtered)

        if self._bad_channels:
            print(f"  Bad channels detected: {self._bad_channels}")
        else:
            print("  No bad channels detected")
