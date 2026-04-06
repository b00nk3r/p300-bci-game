"""
EEG Acquisition Script — g.Nautilus → LSL Stream (via pygds)

Uses the pygds double-buffered GetData callback pattern for
continuous real-time streaming at 500 Hz.

Usage:
    python eeg_acquisition.py
"""

import sys
import time
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

try:
    from src.data.eeg_hdf5_recorder import EEGHDF5Recorder
except ImportError as exc:
    print(exc)
    sys.exit(1)

try:
    import pygds
except ImportError:
    print("pygds not found. Install from:")
    print('  pip install "C:\\Program Files\\gtec\\gNEEDaccess Client API\\Python\\pygds-1.20.1-py3-none-any.whl"')
    sys.exit(1)

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except ImportError:
    print("pylsl not found. Install with: pip install pylsl")
    sys.exit(1)


SAMPLING_RATE = 500
N_CHANNELS = 16
STREAM_NAME = "gNautilus"
STREAM_TYPE = "EEG"
SCAN_COUNT = 64  # Samples per callback (128ms at 500Hz)
HDF5_OUTPUT_DIR = "data/eeg_recordings"

BANDPASS_LOW_HZ = 0.1
BANDPASS_HIGH_HZ = 100.0
BANDPASS_ORDER = 4

NOTCH_LOW_HZ = 48.0
NOTCH_HIGH_HZ = 52.0
NOTCH_ORDER = 4


class RealtimeEEGFilter:
    """Chunk-safe causal filter chain for the acquisition stream."""

    def __init__(self, sampling_rate=SAMPLING_RATE, n_channels=N_CHANNELS):
        nyquist = sampling_rate / 2.0
        if not (0.0 < BANDPASS_LOW_HZ < BANDPASS_HIGH_HZ < nyquist):
            raise ValueError("Bandpass cutoffs must lie strictly inside Nyquist.")
        if not (0.0 < NOTCH_LOW_HZ < NOTCH_HIGH_HZ < nyquist):
            raise ValueError("Notch cutoffs must lie strictly inside Nyquist.")

        self.n_channels = n_channels
        self.bandpass_sos = butter(
            BANDPASS_ORDER,
            [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
            btype="bandpass",
            fs=sampling_rate,
            output="sos",
        )
        self.notch_sos = butter(
            NOTCH_ORDER,
            [NOTCH_LOW_HZ, NOTCH_HIGH_HZ],
            btype="bandstop",
            fs=sampling_rate,
            output="sos",
        )
        self._bandpass_state = None
        self._notch_state = None

    @staticmethod
    def _initial_state(sos, first_sample):
        base_state = sosfilt_zi(sos)[:, :, np.newaxis]
        return base_state * first_sample[np.newaxis, np.newaxis, :]

    def process_chunk(self, samples):
        eeg = np.asarray(samples, dtype=np.float64)
        if eeg.ndim != 2:
            raise ValueError("Expected chunk shape (samples, channels).")
        if eeg.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {eeg.shape[1]}."
            )
        if eeg.shape[0] == 0:
            return eeg.astype(np.float32, copy=False)

        if self._bandpass_state is None:
            self._bandpass_state = self._initial_state(self.bandpass_sos, eeg[0])

        bandpassed, self._bandpass_state = sosfilt(
            self.bandpass_sos,
            eeg,
            axis=0,
            zi=self._bandpass_state,
        )

        if self._notch_state is None:
            self._notch_state = self._initial_state(self.notch_sos, bandpassed[0])

        filtered, self._notch_state = sosfilt(
            self.notch_sos,
            bandpassed,
            axis=0,
            zi=self._notch_state,
        )
        return filtered.astype(np.float32, copy=False)


def main():
    print("=" * 60)
    print("P300 BCI — EEG Acquisition (g.Nautilus → LSL)")
    print("=" * 60)

    # ── Find and connect ─────────────────────────────────────────────
    print("\nSearching for connected devices...")
    devices = pygds.ConnectedDevices()

    if not devices:
        print("ERROR: No g.tec devices found.")
        sys.exit(1)

    for serial, dev_type, in_use in devices:
        type_name = {1: "g.HIamp", 2: "g.USBamp", 3: "g.Nautilus"}.get(dev_type, f"Unknown({dev_type})")
        status = "IN USE" if in_use else "available"
        print(f"  {serial} — {type_name} ({status})")

    d = pygds.GDS()
    print(f"\nConnected to {d.Name}")
    print(f"  Sample rate: {d.SamplingRate} Hz")
    print(f"  Electrodes: {d.N_electrodes}")

    if abs(float(d.SamplingRate) - SAMPLING_RATE) > 1e-6:
        print(
            f"ERROR: Device sample rate is {d.SamplingRate} Hz, "
            f"but this app expects {SAMPLING_RATE} Hz."
        )
        print("Update the g.Nautilus configuration to 500 Hz before streaming.")
        sys.exit(1)

    if int(d.N_electrodes) < N_CHANNELS:
        print(
            f"ERROR: Device exposes {d.N_electrodes} electrodes, "
            f"but the game expects at least {N_CHANNELS}."
        )
        sys.exit(1)

    stream_filter = RealtimeEEGFilter(
        sampling_rate=SAMPLING_RATE,
        n_channels=N_CHANNELS,
    )

    # ── Create LSL stream ────────────────────────────────────────────
    lsl_info = StreamInfo(
        name=STREAM_NAME,
        type=STREAM_TYPE,
        channel_count=N_CHANNELS,
        nominal_srate=SAMPLING_RATE,
        channel_format="float32",
        source_id=f"gnautilus_{d.Name}",
    )
    outlet = StreamOutlet(lsl_info)

    print(f"\nLSL stream: {STREAM_NAME} ({N_CHANNELS} ch @ {SAMPLING_RATE} Hz)")
    print(
        "Acquisition preprocessing: "
        f"bandpass {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz, "
        f"notch {NOTCH_LOW_HZ}-{NOTCH_HIGH_HZ} Hz"
    )
    print(f"Press Ctrl+C to stop.\n")

    session_start_unix_s = time.time()
    session_start_unix_ns = time.time_ns()
    session_start_lsl_s = local_clock()
    unix_lsl_offset_s = session_start_unix_s - session_start_lsl_s

    recorder = EEGHDF5Recorder(output_dir=HDF5_OUTPUT_DIR)
    recording_path = recorder.start(
        sampling_rate=SAMPLING_RATE,
        n_channels=N_CHANNELS,
        stream_name=STREAM_NAME,
        stream_type=STREAM_TYPE,
        device_name=str(d.Name),
        scan_count=SCAN_COUNT,
        bandpass_low_hz=BANDPASS_LOW_HZ,
        bandpass_high_hz=BANDPASS_HIGH_HZ,
        bandpass_order=BANDPASS_ORDER,
        notch_low_hz=NOTCH_LOW_HZ,
        notch_high_hz=NOTCH_HIGH_HZ,
        notch_order=NOTCH_ORDER,
        session_start_unix_s=session_start_unix_s,
        session_start_unix_ns=session_start_unix_ns,
        session_start_lsl_s=session_start_lsl_s,
        unix_lsl_offset_s=unix_lsl_offset_s,
        timestamp_reference=(
            "Per-sample LSL timestamps are reconstructed from acquisition callback "
            "arrival time and nominal sampling rate; unix_timestamps use the "
            "session-start unix-minus-lsl offset."
        ),
    )
    print(f"HDF5 recording: {recording_path}\n")

    # ── Streaming state ──────────────────────────────────────────────
    running = True
    samples_sent = 0
    start_time = session_start_unix_s
    last_report = start_time

    def on_data(samples):
        """
        Callback invoked by GetData for each chunk of samples.
        Pushes EEG data to LSL and returns True to keep streaming.
        """
        nonlocal samples_sent, last_report, running

        if not running:
            return False  # Stop acquisition

        # samples is (scan_count, n_channels) float32 array
        eeg = samples[:, :N_CHANNELS] if samples.shape[1] > N_CHANNELS else samples

        filtered_eeg = stream_filter.process_chunk(eeg)

        chunk_sample_count = filtered_eeg.shape[0]
        if chunk_sample_count == 0:
            return True

        sample_period_s = 1.0 / SAMPLING_RATE
        chunk_end_lsl_s = local_clock()
        chunk_start_lsl_s = chunk_end_lsl_s - ((chunk_sample_count - 1) * sample_period_s)
        chunk_lsl_timestamps = chunk_start_lsl_s + (
            np.arange(chunk_sample_count, dtype=np.float64) * sample_period_s
        )
        chunk_unix_timestamps = chunk_lsl_timestamps + unix_lsl_offset_s

        recorder.append_chunk(filtered_eeg, chunk_lsl_timestamps, chunk_unix_timestamps)

        # Push to LSL sample by sample after acquisition-side filtering
        for i in range(filtered_eeg.shape[0]):
            outlet.push_sample(
                filtered_eeg[i].tolist(),
                timestamp=float(chunk_lsl_timestamps[i]),
            )

        samples_sent += eeg.shape[0]

        # Report every 10 seconds
        now = time.time()
        if now - last_report >= 10.0:
            elapsed = now - start_time
            rate = samples_sent / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.0f}s] {samples_sent} samples ({rate:.0f} Hz)")
            last_report = now

        return True  # Keep streaming

    # ── Run acquisition (blocking call with callback) ────────────────
    print("Streaming... Start the BCI game now.\n")

    try:
        # GetData blocks and continuously calls on_data with chunks
        # of SCAN_COUNT samples until on_data returns False
        d.GetData(SCAN_COUNT, more=on_data)
    except KeyboardInterrupt:
        running = False
        elapsed = time.time() - start_time
        print(f"\nStopped after {elapsed:.1f}s, {samples_sent} samples.")
    finally:
        saved_path = recorder.close()
        if saved_path:
            print(f"HDF5 saved: {saved_path}")

    # ── Cleanup ──────────────────────────────────────────────────────
    print("Closing device...")
    try:
        d.Close()
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()