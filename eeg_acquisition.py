"""
EEG Acquisition Script — g.Nautilus → LSL Stream

Run this on the Windows lab PC where the g.Nautilus base station
is connected via USB. It acquires 16-channel EEG at 500 Hz and
broadcasts it as an LSL stream that the game can pick up.

This script uses g.Pype (g.tec's Python SDK). Install it following
the g.Pype Training Season 1 Episode 1 instructions.

Usage:
    python eeg_acquisition.py

The game's RealtimeClassifier will automatically discover this
LSL stream and start receiving data.

References:
    - g.Pype Training S3E2: g.Nautilus setup
    - g.Pype Training S4E2: Sending LSL streams
    - https://gpype.gtec.at/
"""

import gpype as gp

# ── Configuration ────────────────────────────────────────────────────

SAMPLING_RATE = 500   # Hz — must match the classifier's SR constant
N_CHANNELS = 16       # 16-channel g.Nautilus headset

# g.Nautilus serial number (printed on both headset and base station).
# Replace with YOUR device's serial number.
SERIAL_NUMBER = "NR-XXXX.XX.XX"

# Bandpass filter applied in hardware (before LSL transmission).
# The classifier also applies its own software bandpass, so this is
# an optional first pass for noise reduction during transmission.
# Set to None to send raw unfiltered data.
HARDWARE_BANDPASS = (0.5, 100.0)  # Hz — wide pass, let software do the rest

# Notch filter for power line interference (60 Hz in US)
HARDWARE_NOTCH = (58.0, 62.0)  # Hz — set to (48, 52) for 50 Hz countries


if __name__ == "__main__":
    print("=" * 60)
    print("P300 BCI — EEG Acquisition (g.Nautilus → LSL)")
    print("=" * 60)

    # Create the processing pipeline
    pipeline = gp.Pipeline()

    # ── g.Nautilus data source ───────────────────────────────────────
    # This connects to your 16-channel g.Nautilus headset via the
    # base station. The headset must be powered on and paired.
    #
    # If you don't know the serial, comment out serial_number and
    # g.Pype will try to auto-detect the connected device.
    nautilus = gp.GNautilus(
        sampling_rate=SAMPLING_RATE,
        # serial_number=SERIAL_NUMBER,  # Uncomment and set your serial
    )

    # ── Optional: hardware-level bandpass filter ─────────────────────
    if HARDWARE_BANDPASS:
        bandpass = gp.BandpassFilter(
            low_frequency=HARDWARE_BANDPASS[0],
            high_frequency=HARDWARE_BANDPASS[1],
        )

    # ── Optional: hardware-level notch filter ────────────────────────
    if HARDWARE_NOTCH:
        notch = gp.NotchFilter(
            low_frequency=HARDWARE_NOTCH[0],
            high_frequency=HARDWARE_NOTCH[1],
        )

    # ── LSL output stream ────────────────────────────────────────────
    # This makes the EEG data discoverable on the local network.
    # The game's LSLReceiver will find it by stream type "EEG".
    lsl_sender = gp.LSLSender()

    # ── Connect the pipeline ─────────────────────────────────────────
    if HARDWARE_BANDPASS and HARDWARE_NOTCH:
        pipeline.connect(nautilus, bandpass)
        pipeline.connect(bandpass, notch)
        pipeline.connect(notch, lsl_sender)
    elif HARDWARE_BANDPASS:
        pipeline.connect(nautilus, bandpass)
        pipeline.connect(bandpass, lsl_sender)
    else:
        pipeline.connect(nautilus, lsl_sender)

    # ── Run ──────────────────────────────────────────────────────────
    print(f"\nDevice:       g.Nautilus ({N_CHANNELS} channels)")
    print(f"Sample rate:  {SAMPLING_RATE} Hz")
    print(f"Output:       LSL stream (type='EEG')")
    print(f"\nStarting pipeline...")

    pipeline.start()

    print("Pipeline is running. LSL stream is broadcasting.")
    print("Start the BCI game on this or another computer.")
    print("\nPress Enter to stop.\n")

    input()

    pipeline.stop()
    print("Pipeline stopped. EEG stream closed.")
