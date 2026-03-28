"""
EEG Acquisition Script — g.Nautilus → LSL Stream (via pygds)

Uses the pygds double-buffered GetData callback pattern for
continuous real-time streaming at 500 Hz.

Usage:
    python eeg_acquisition.py
"""

import sys
import time
import threading
import numpy as np

try:
    import pygds
except ImportError:
    print("pygds not found. Install from:")
    print('  pip install "C:\\Program Files\\gtec\\gNEEDaccess Client API\\Python\\pygds-1.20.1-py3-none-any.whl"')
    sys.exit(1)

try:
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    print("pylsl not found. Install with: pip install pylsl")
    sys.exit(1)


SAMPLING_RATE = 500
N_CHANNELS = 16
STREAM_NAME = "gNautilus"
STREAM_TYPE = "EEG"
SCAN_COUNT = 64  # Samples per callback (128ms at 500Hz)


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
    print(f"Press Ctrl+C to stop.\n")

    # ── Streaming state ──────────────────────────────────────────────
    running = True
    samples_sent = 0
    start_time = time.time()
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

        # Push to LSL sample by sample
        for i in range(eeg.shape[0]):
            outlet.push_sample(eeg[i].tolist())

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

    # ── Cleanup ──────────────────────────────────────────────────────
    print("Closing device...")
    try:
        d.Close()
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()