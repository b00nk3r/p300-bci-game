#!/usr/bin/env python3
"""
Mock EEG Stream
===============
Simulates a 16-channel, 500 Hz EEG stream over LSL for development
and testing without actual EEG hardware.

Usage:
    python mock_eeg_stream.py

Then run the game in another terminal — it will auto-connect to this
stream.  Classification results will be random (the data is noise),
but this exercises the entire pipeline end-to-end.
"""

import time
import numpy as np

from pylsl import StreamInfo, StreamOutlet

CHANNELS = 16
SRATE = 500
STREAM_NAME = "MockEEG"
STREAM_TYPE = "EEG"
SOURCE_ID = "mock_eeg_001"


def main():
    info = StreamInfo(
        STREAM_NAME, STREAM_TYPE, CHANNELS, SRATE, "float32", SOURCE_ID
    )
    outlet = StreamOutlet(info)

    print(f"Mock EEG stream started: {CHANNELS}ch @ {SRATE}Hz")
    print("Press Ctrl+C to stop.\n")

    sample_interval = 1.0 / SRATE
    next_send = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            if now >= next_send:
                sample = (np.random.randn(CHANNELS) * 10).tolist()
                outlet.push_sample(sample)
                next_send += sample_interval
                # Catch up if we fell behind
                if next_send < now - 1.0:
                    next_send = now
            else:
                time.sleep(max(0, next_send - now - 0.0002))
    except KeyboardInterrupt:
        print("\nMock stream stopped.")


if __name__ == "__main__":
    main()
