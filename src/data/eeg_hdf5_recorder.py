"""
Continuous HDF5 recorder for filtered EEG acquisition data.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover - exercised in runtime environments
    raise ImportError(
        "h5py is required for HDF5 EEG recording. Install it with: pip install h5py"
    ) from exc


class EEGHDF5Recorder:
    """Append filtered EEG samples and timestamps to a continuous HDF5 file."""

    def __init__(self, output_dir: str = "data/eeg_recordings", flush_interval_s: float = 10.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval_s = flush_interval_s

        self._file: Optional[h5py.File] = None
        self._samples_ds = None
        self._lsl_timestamps_ds = None
        self._unix_timestamps_ds = None
        self._sample_index_ds = None
        self._last_flush_unix_s = 0.0
        self._sample_count = 0
        self._filepath: Optional[Path] = None

    def start(
        self,
        *,
        sampling_rate: float,
        n_channels: int,
        stream_name: str,
        stream_type: str,
        device_name: str,
        scan_count: int,
        bandpass_low_hz: float,
        bandpass_high_hz: float,
        bandpass_order: int,
        notch_low_hz: float,
        notch_high_hz: float,
        notch_order: int,
        session_start_unix_s: float,
        session_start_unix_ns: int,
        session_start_lsl_s: float,
        unix_lsl_offset_s: float,
        timestamp_reference: str,
    ) -> str:
        """Create a new HDF5 recording file and write session metadata."""
        if self._file is not None:
            raise RuntimeError("Recorder already started.")

        start_dt = datetime.fromtimestamp(session_start_unix_s)
        session_id = start_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._filepath = self.output_dir / f"eeg_recording_{session_id}.h5"

        self._file = h5py.File(self._filepath, "w")
        self._file.attrs["session_id"] = session_id
        self._file.attrs["session_start_datetime"] = start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._file.attrs["session_start_unix_s"] = float(session_start_unix_s)
        self._file.attrs["session_start_unix_ns"] = int(session_start_unix_ns)
        self._file.attrs["session_start_lsl_s"] = float(session_start_lsl_s)
        self._file.attrs["unix_lsl_offset_s"] = float(unix_lsl_offset_s)
        self._file.attrs["timestamp_reference"] = timestamp_reference

        self._file.attrs["stream_name"] = stream_name
        self._file.attrs["stream_type"] = stream_type
        self._file.attrs["device_name"] = device_name
        self._file.attrs["sampling_rate_hz"] = float(sampling_rate)
        self._file.attrs["n_channels"] = int(n_channels)
        self._file.attrs["scan_count"] = int(scan_count)

        self._file.attrs["bandpass_low_hz"] = float(bandpass_low_hz)
        self._file.attrs["bandpass_high_hz"] = float(bandpass_high_hz)
        self._file.attrs["bandpass_order"] = int(bandpass_order)
        self._file.attrs["notch_low_hz"] = float(notch_low_hz)
        self._file.attrs["notch_high_hz"] = float(notch_high_hz)
        self._file.attrs["notch_order"] = int(notch_order)

        chunk_rows = max(int(scan_count), 1)
        self._samples_ds = self._file.create_dataset(
            "samples",
            shape=(0, n_channels),
            maxshape=(None, n_channels),
            chunks=(chunk_rows, n_channels),
            dtype=np.float32,
        )
        self._lsl_timestamps_ds = self._file.create_dataset(
            "lsl_timestamps",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_rows,),
            dtype=np.float64,
        )
        self._unix_timestamps_ds = self._file.create_dataset(
            "unix_timestamps",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_rows,),
            dtype=np.float64,
        )
        self._sample_index_ds = self._file.create_dataset(
            "sample_index",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_rows,),
            dtype=np.int64,
        )

        self._last_flush_unix_s = session_start_unix_s
        self._sample_count = 0
        self._file.flush()
        return str(self._filepath)

    def append_chunk(self, samples, lsl_timestamps, unix_timestamps):
        """Append one EEG chunk and matching timestamps."""
        if self._file is None:
            raise RuntimeError("Recorder has not been started.")

        samples_np = np.asarray(samples, dtype=np.float32)
        lsl_np = np.asarray(lsl_timestamps, dtype=np.float64)
        unix_np = np.asarray(unix_timestamps, dtype=np.float64)

        if samples_np.ndim != 2:
            raise ValueError("samples must have shape (n_samples, n_channels).")
        if lsl_np.ndim != 1 or unix_np.ndim != 1:
            raise ValueError("timestamps must be 1-D arrays.")
        if samples_np.shape[1] != self._samples_ds.shape[1]:
            raise ValueError(
                f"Expected {self._samples_ds.shape[1]} channels, got {samples_np.shape[1]}."
            )
        if samples_np.shape[0] != lsl_np.shape[0] or samples_np.shape[0] != unix_np.shape[0]:
            raise ValueError("samples and timestamps must have matching lengths.")
        if samples_np.shape[0] == 0:
            return

        start = self._sample_count
        end = start + samples_np.shape[0]

        self._samples_ds.resize((end, samples_np.shape[1]))
        self._lsl_timestamps_ds.resize((end,))
        self._unix_timestamps_ds.resize((end,))
        self._sample_index_ds.resize((end,))

        self._samples_ds[start:end] = samples_np
        self._lsl_timestamps_ds[start:end] = lsl_np
        self._unix_timestamps_ds[start:end] = unix_np
        self._sample_index_ds[start:end] = np.arange(start, end, dtype=np.int64)

        self._sample_count = end
        self._maybe_flush()

    def _maybe_flush(self, force: bool = False):
        """Flush periodically so long sessions are persisted during acquisition."""
        if self._file is None:
            return

        now = time.time()
        if force or (now - self._last_flush_unix_s) >= self.flush_interval_s:
            self._file.flush()
            self._last_flush_unix_s = now

    def close(
        self,
        *,
        recording_end_unix_s: Optional[float] = None,
        recording_end_unix_ns: Optional[int] = None,
    ) -> Optional[str]:
        """Finalize the HDF5 file and write summary metadata."""
        if self._file is None:
            return str(self._filepath) if self._filepath else None

        if recording_end_unix_s is None:
            recording_end_unix_s = time.time()
        if recording_end_unix_ns is None:
            recording_end_unix_ns = time.time_ns()

        self._file.attrs["recording_end_datetime"] = datetime.fromtimestamp(
            recording_end_unix_s
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._file.attrs["recording_end_unix_s"] = float(recording_end_unix_s)
        self._file.attrs["recording_end_unix_ns"] = int(recording_end_unix_ns)
        self._file.attrs["duration_s"] = float(
            recording_end_unix_s - self._file.attrs["session_start_unix_s"]
        )
        self._file.attrs["n_samples_written"] = int(self._sample_count)

        self._maybe_flush(force=True)
        self._file.close()
        self._file = None
        self._samples_ds = None
        self._lsl_timestamps_ds = None
        self._unix_timestamps_ds = None
        self._sample_index_ds = None
        return str(self._filepath) if self._filepath else None

    @property
    def filepath(self) -> Optional[str]:
        """Return the current output path, if any."""
        return str(self._filepath) if self._filepath else None
