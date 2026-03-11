"""
EEG Receiver
============
Connects to an LSL EEG stream (e.g. from g.Recorder) and continuously
buffers incoming samples in a thread-safe circular buffer.

Provides:
- Background daemon thread for continuous sample acquisition
- Thread-safe get_chunk() to retrieve EEG data between two LSL timestamps
- get_current_time() returning pylsl.local_clock() for flash-event stamping
"""

import threading
import numpy as np

from bci_config import SR, N_CHANNELS, BUFFER_DURATION_S

try:
    import pylsl
    HAS_PYLSL = True
except ImportError:
    HAS_PYLSL = False


class EEGReceiver:
    """
    Receives EEG data over LSL and stores it in a circular buffer.

    The background thread calls inlet.pull_chunk() in a loop and writes
    samples + timestamps into a fixed-size numpy ring buffer protected
    by a threading.Lock.
    """

    def __init__(self, buffer_duration_s: int = BUFFER_DURATION_S):
        if not HAS_PYLSL:
            raise ImportError(
                "pylsl is not installed. Install with: pip install pylsl"
            )

        self._buffer_size = SR * buffer_duration_s
        self._data = np.zeros((self._buffer_size, N_CHANNELS), dtype=np.float64)
        self._timestamps = np.zeros(self._buffer_size, dtype=np.float64)
        self._write_pos = 0
        self._samples_written = 0

        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._inlet = None
        self._connected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, timeout: float = 10.0):
        """Resolve the LSL EEG stream and start the acquisition thread.

        Args:
            timeout: Max seconds to wait for a stream to appear.

        Raises:
            RuntimeError: If no EEG stream is found within *timeout*.
        """
        streams = pylsl.resolve_stream("type", "EEG", 1, timeout)
        if not streams:
            raise RuntimeError(
                f"No LSL EEG stream found within {timeout}s. "
                "Make sure g.Recorder (or a mock stream) is running."
            )

        self._inlet = pylsl.StreamInlet(
            streams[0],
            max_buflen=BUFFER_DURATION_S,
            max_chunklen=0,
        )
        self._inlet.open_stream()

        self._running = True
        self._connected = True
        self._thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the acquisition thread and close the inlet."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None
        self._connected = False

    def is_connected(self) -> bool:
        """Return True if the receiver is actively collecting data."""
        return self._connected and self._running

    @staticmethod
    def get_current_time() -> float:
        """Return the current LSL clock value.

        The game should call this (instead of time.perf_counter()) when
        timestamping flash events so that flash times live in the same
        clock domain as the EEG sample timestamps.
        """
        return pylsl.local_clock()

    def get_chunk(self, start_time: float, end_time: float):
        """Retrieve buffered EEG data between two LSL timestamps.

        Args:
            start_time: Earliest LSL timestamp to include.
            end_time:   Latest LSL timestamp to include.

        Returns:
            data:       numpy array (n_samples, n_channels)
            timestamps: numpy array (n_samples,)
            Both are copies, safe to mutate.
        """
        with self._lock:
            n = min(self._samples_written, self._buffer_size)
            if n == 0:
                return (
                    np.empty((0, N_CHANNELS), dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )

            if self._samples_written <= self._buffer_size:
                ts = self._timestamps[:n]
                dat = self._data[:n]
            else:
                # Unwrap the circular buffer into chronological order
                idx = np.concatenate([
                    np.arange(self._write_pos, self._buffer_size),
                    np.arange(0, self._write_pos),
                ])
                ts = self._timestamps[idx]
                dat = self._data[idx]

            mask = (ts >= start_time) & (ts <= end_time)
            return dat[mask].copy(), ts[mask].copy()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _acquire_loop(self):
        """Continuously pull samples from the LSL inlet into the buffer."""
        while self._running:
            try:
                samples, timestamps = self._inlet.pull_chunk(
                    timeout=0.1, max_samples=256
                )
            except Exception:
                continue

            if not timestamps:
                continue

            chunk = np.array(samples, dtype=np.float64)
            ts = np.array(timestamps, dtype=np.float64)
            n = len(ts)

            with self._lock:
                end = self._write_pos + n
                if end <= self._buffer_size:
                    self._data[self._write_pos:end] = chunk
                    self._timestamps[self._write_pos:end] = ts
                else:
                    first = self._buffer_size - self._write_pos
                    self._data[self._write_pos:] = chunk[:first]
                    self._timestamps[self._write_pos:] = ts[:first]
                    remainder = n - first
                    self._data[:remainder] = chunk[first:]
                    self._timestamps[:remainder] = ts[first:]

                self._write_pos = end % self._buffer_size
                self._samples_written += n
