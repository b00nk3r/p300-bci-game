"""
Mock LSL Stream — for testing without EEG hardware

Creates a synthetic 16-channel LSL stream at 500 Hz that mimics
the g.Nautilus output. Use this to test the game + classifier
integration on your MacBook before going to the lab.

The signal is random noise with optional injected P300-like bumps.
It won't produce meaningful classifications, but it validates that:
  - The LSL connection works
  - The ring buffer fills correctly
  - Epoch extraction timing is correct
  - The preprocessing pipeline runs without errors
  - The model produces outputs (even if wrong)

Usage:
    Terminal 1:  python mock_lsl_stream.py
    Terminal 2:  python main_live.py
"""

import numpy as np
import time

try:
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    raise ImportError("Install pylsl: pip install pylsl")

from config import EEG_SAMPLING_RATE_HZ as SR, EEG_N_CHANNELS as N_CHANNELS

STREAM_NAME = "MockEEG"
STREAM_TYPE = "EEG"


def main():
    print(f"Creating mock LSL stream: {STREAM_NAME}")
    print(f"  Channels: {N_CHANNELS}")
    print(f"  Sample rate: {SR} Hz")
    print(f"  Type: {STREAM_TYPE}")

    info = StreamInfo(
        name=STREAM_NAME,
        type=STREAM_TYPE,
        channel_count=N_CHANNELS,
        nominal_srate=SR,
        channel_format="double64",
        source_id="mock_eeg_source",
    )

    outlet = StreamOutlet(info)
    print(f"\nStream is broadcasting. Ctrl+C to stop.\n")

    samples_sent = 0
    start_time = time.time()

    try:
        while True:
            # Generate a chunk of 10 samples at a time (20ms worth)
            chunk_size = 10
            chunk = np.random.randn(chunk_size, N_CHANNELS) * 15.0  # ~15 µV noise

            # Add a subtle 10 Hz alpha rhythm on posterior channels (12-15)
            t = np.arange(chunk_size) / SR + samples_sent / SR
            alpha = 8.0 * np.sin(2 * np.pi * 10 * t)
            for ch in range(12, N_CHANNELS):
                chunk[:, ch] += alpha

            for i in range(chunk_size):
                outlet.push_sample(chunk[i].tolist())

            samples_sent += chunk_size

            # Print status every 5 seconds
            elapsed = time.time() - start_time
            if samples_sent % (SR * 5) == 0:
                effective_rate = samples_sent / elapsed if elapsed > 0 else 0
                print(f"  Sent {samples_sent} samples "
                      f"({elapsed:.1f}s, {effective_rate:.0f} Hz effective)")

            # Sleep to maintain approximate real-time rate
            # (LSL handles precise timing, but we avoid spinning the CPU)
            time.sleep(chunk_size / SR * 0.9)

    except KeyboardInterrupt:
        print(f"\nStopped. Sent {samples_sent} samples total.")


if __name__ == "__main__":
    main()
