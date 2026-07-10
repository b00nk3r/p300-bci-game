# p300-bci-game

A P300 brain-computer interface game, built as a Salem State University
undergraduate computer science capstone project.

The player collects donuts in a 2D pixel-art maze by attending to arrows that
flash in the center of the screen. Attending to a flashing arrow produces a
P300 response in the EEG, which a trained classifier decodes into a movement.

## How it works

The project has two modes, each with its own launcher:

- **Calibration** (`main_calibration.py`) — runs 12 flashing phases (3 shuffled
  blocks of the 4 directions), showing an ATTEND instruction before each. EEG is
  recorded during this mode, alongside a trigger log and a session log holding
  the flash timing needed to train the model.
- **Live** (`main_live.py`) — SPACE starts a selection; the arrows flash, the
  classifier decodes the attended direction from the live EEG, the player moves,
  and after a short break the next selection begins. Collecting all 5 donuts wins
  the game. Live mode does not record EEG — each sample is fed straight to the
  classifier.

Model training happens separately (in a notebook in another repository). The
trained `.joblib` model is copied into `models/` and referenced by `MODEL_PATH`
in `config.py`.

## Pipeline

1. **Acquire** — `eeg_acquisition.py` streams the g.Nautilus device to an LSL
   stream (and records raw EEG to HDF5).
2. **Calibrate** — run `main_calibration.py` to collect labeled flash data.
3. **Train** — train the model offline, then drop the `.joblib` into `models/`
   and point `MODEL_PATH` at it.
4. **Play** — run `main_live.py` with the EEG stream live.

## Running

```bash
pip install -r requirements.txt

python main_calibration.py     # calibration / data collection
python main_live.py            # live BCI gameplay
```

Both launchers accept `--fullscreen`, `--width`, `--height`, `--display`, and
`--sequences` (see `python main_live.py --help`). In-game: **SPACE** starts a
run, **S** opens settings, **R** restarts, **ESC** quits.

## Testing without hardware

`mock_lsl_stream.py` publishes a synthetic 16-channel LSL stream so the game and
classifier can be exercised without a headset. Its output is noise, so
classifications are meaningless — it only validates the LSL connection, buffering,
epoching, and preprocessing.

```bash
python mock_lsl_stream.py      # terminal 1
python main_live.py            # terminal 2
```

If no EEG stream is found, live mode falls back to a mock classifier that returns
random directions.
