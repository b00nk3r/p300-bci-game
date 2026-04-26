"""
Calibration training pipeline.

Glues together the live game's recorded calibration data and the offline
training utilities under ``TOBE_INTEGRATED``:

    1. Find the most recent EEG HDF5 recording (saved by ``eeg_acquisition.py``).
    2. Pick the most recent N session/trigger pairs (saved by ``main.py``).
    3. Copy them into ``data/test/{eeg,sessions}`` (the layout expected
       by ``TOBE_INTEGRATED/preprocess_test_epochs.py``).
    4. Run preprocessing -> ``data/test_processed/clean_epochs.npz``.
    5. Train the single-trial LDA -> ``models/10trials_model.joblib``.

The script can be invoked as a CLI or as a Python function so the game
loop can launch it from a background thread.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_EEG_DIR = PROJECT_ROOT / "data" / "eeg_recordings"
DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_TEST_PROCESSED_DIR = PROJECT_ROOT / "data" / "test_processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "10trials_model.joblib"
DEFAULT_N_RUNS = 10


_TIMESTAMP_PATTERN = re.compile(r"(\d{8})_(\d{6})")


def _parse_timestamp(name: str) -> Optional[datetime]:
    match = _TIMESTAMP_PATTERN.search(name)
    if not match:
        return None
    d, t = match.group(1), match.group(2)
    try:
        return datetime(
            int(d[:4]), int(d[4:6]), int(d[6:8]),
            int(t[:2]), int(t[2:4]), int(t[4:6]),
        )
    except ValueError:
        return None


def find_latest_eeg_recording(eeg_dir: Path) -> Path:
    """Return the most recent ``eeg_recording_*.h5`` file."""
    candidates = sorted(eeg_dir.glob("eeg_recording_*.h5"))
    if not candidates:
        raise RuntimeError(
            f"No EEG recordings found under {eeg_dir}. "
            "Run eeg_acquisition.py before starting calibration."
        )
    return candidates[-1]


def wait_for_eeg_recording_ready(
    eeg_path: Path,
    timeout_s: float = 60.0,
    poll_s: float = 1.0,
) -> None:
    """Wait until the H5 recording looks closed and stable enough to copy."""
    deadline = time.perf_counter() + timeout_s
    last_size = -1
    stable_count = 0

    while time.perf_counter() < deadline:
        try:
            current_size = eeg_path.stat().st_size
            if current_size == last_size:
                stable_count += 1
            else:
                stable_count = 0
                last_size = current_size

            # Opening the file through h5py fails while the writer still has
            # the HDF5 metadata in an inconsistent/incomplete state.
            import h5py

            with h5py.File(eeg_path, "r") as f:
                has_required_data = (
                    "samples" in f
                    and "unix_timestamps" in f
                    and f["samples"].shape[0] > 0
                    and f["unix_timestamps"].shape[0] > 0
                )

            if has_required_data and stable_count >= 1:
                return
        except OSError:
            # Expected while the acquisition process is still finalizing.
            pass

        time.sleep(poll_s)

    raise RuntimeError(
        f"Timed out waiting for EEG recording to finish closing: {eeg_path}"
    )


def collect_recent_session_pairs(
    sessions_dir: Path,
    n_runs: int,
    after: Optional[datetime] = None,
) -> List[Tuple[Path, Path, datetime]]:
    """
    Return up to ``n_runs`` (session_path, trigger_path, timestamp) tuples
    sorted from oldest to newest, optionally filtered to entries strictly
    after a reference timestamp.
    """
    sessions = {}
    for p in sessions_dir.glob("session_*.txt"):
        ts = _parse_timestamp(p.name)
        if ts is None:
            continue
        if after is not None and ts < after:
            continue
        sessions[ts] = p

    triggers = {}
    for p in sessions_dir.glob("triggers_*.txt"):
        ts = _parse_timestamp(p.name)
        if ts is None:
            continue
        if after is not None and ts < after:
            continue
        triggers[ts] = p

    common = sorted(set(sessions) & set(triggers))
    if len(common) < n_runs:
        raise RuntimeError(
            f"Only found {len(common)} matched session/trigger pairs in "
            f"{sessions_dir} (need {n_runs}). "
            "Make sure calibration finished before training."
        )

    selected = common[-n_runs:]
    return [(sessions[ts], triggers[ts], ts) for ts in selected]


def stage_test_data(
    eeg_path: Path,
    session_pairs: List[Tuple[Path, Path, datetime]],
    test_dir: Path,
) -> None:
    """
    Copy EEG and session/trigger files into ``test_dir`` so the offline
    preprocessing script can pick them up.

    The destination is wiped first so each calibration produces a clean
    training set.
    """
    eeg_dest = test_dir / "eeg"
    sessions_dest = test_dir / "sessions"

    if eeg_dest.exists():
        shutil.rmtree(eeg_dest)
    if sessions_dest.exists():
        shutil.rmtree(sessions_dest)

    eeg_dest.mkdir(parents=True, exist_ok=True)
    sessions_dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(eeg_path, eeg_dest / eeg_path.name)
    for session_path, trigger_path, _ts in session_pairs:
        shutil.copy2(session_path, sessions_dest / session_path.name)
        shutil.copy2(trigger_path, sessions_dest / trigger_path.name)

    print(
        f"Staged {len(session_pairs)} session pairs + 1 EEG recording "
        f"to {test_dir}"
    )


def run_pipeline(
    n_runs: int = DEFAULT_N_RUNS,
    eeg_dir: Path = DEFAULT_EEG_DIR,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    test_processed_dir: Path = DEFAULT_TEST_PROCESSED_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    after: Optional[datetime] = None,
) -> Path:
    """
    Run the complete calibration -> preprocess -> train pipeline.

    Returns the path of the trained model artifact.
    """
    eeg_dir = Path(eeg_dir)
    sessions_dir = Path(sessions_dir)
    test_dir = Path(test_dir)
    test_processed_dir = Path(test_processed_dir)
    model_path = Path(model_path)

    print("=" * 60)
    print("Calibration training pipeline")
    print("=" * 60)
    t0 = time.perf_counter()

    eeg_path = find_latest_eeg_recording(eeg_dir)
    print(f"EEG recording: {eeg_path.name}")
    print("Waiting for EEG recording to finish closing...")
    wait_for_eeg_recording_ready(eeg_path)

    session_pairs = collect_recent_session_pairs(
        sessions_dir, n_runs=n_runs, after=after
    )
    print(f"Selected {len(session_pairs)} calibration runs:")
    for s, t, ts in session_pairs:
        print(f"  {ts}  {s.name}")

    stage_test_data(eeg_path, session_pairs, test_dir)

    sys.path.insert(0, str(PROJECT_ROOT / "TOBE_INTEGRATED"))
    from preprocess_test_epochs import run_preprocessing
    from train_lda_10 import train_model

    npz_path = run_preprocessing(test_dir, test_processed_dir)
    saved_model = train_model(npz_path, model_path, n_runs=n_runs)

    elapsed = time.perf_counter() - t0
    print(f"\nPipeline finished in {elapsed:.1f}s")
    return saved_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--eeg-dir", type=Path, default=DEFAULT_EEG_DIR)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--test-processed-dir", type=Path, default=DEFAULT_TEST_PROCESSED_DIR
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Only consider sessions newer than this UTC timestamp "
             "(YYYYmmdd_HHMMSS).",
    )
    args = parser.parse_args()

    after_ts = _parse_timestamp(args.after) if args.after else None

    try:
        run_pipeline(
            n_runs=args.n_runs,
            eeg_dir=args.eeg_dir,
            sessions_dir=args.sessions_dir,
            test_dir=args.test_dir,
            test_processed_dir=args.test_processed_dir,
            model_path=args.model_path,
            after=after_ts,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
