"""
Preprocess calibration epochs collected by the live game.

This is a path-aware version of the original TOBE_INTEGRATED preprocessing
script. It can be run from any working directory because all paths are
resolved relative to the project root.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py


PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from .config import (
        SR,
        EPOCH_PRE_MS,
        EPOCH_POST_MS,
        DIRECTION_MAP,
    )
    from .utils import (
        parse_trigger_file,
        compute_recording_offset,
        extract_epochs,
        apply_bandpass_filter,
        detect_bad_channels,
        apply_car,
        apply_baseline_correction,
        reject_artifacts,
    )
except ImportError:  # Allow direct execution: python TOBE_INTEGRATED/preprocess_test_epochs.py
    from config import (  # type: ignore
        SR,
        EPOCH_PRE_MS,
        EPOCH_POST_MS,
        DIRECTION_MAP,
    )
    from utils import (  # type: ignore
        parse_trigger_file,
        compute_recording_offset,
        extract_epochs,
        apply_bandpass_filter,
        detect_bad_channels,
        apply_car,
        apply_baseline_correction,
        reject_artifacts,
    )


DEFAULT_TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "test_processed"


def parse_eeg_timestamp(filename: str):
    match = re.search(r"eeg_recording_(\d{8})_(\d{6})", filename)
    if match:
        d, t = match.group(1), match.group(2)
        return datetime(
            int(d[:4]), int(d[4:6]), int(d[6:8]),
            int(t[:2]), int(t[2:4]), int(t[4:6]),
        )
    return None


def parse_txt_timestamp(filename: str):
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        d, t = match.group(1), match.group(2)
        return datetime(
            int(d[:4]), int(d[4:6]), int(d[6:8]),
            int(t[:2]), int(t[2:4]), int(t[4:6]),
        )
    return None


def load_test_eeg(path: Path):
    """Load EEG from H5 format (flat structure with unix_timestamps)."""
    with h5py.File(path, "r") as f:
        eeg = f["samples"][:]
        recording_start_unix = float(f["unix_timestamps"][0])
    return eeg, recording_start_unix


def parse_session_result(session_path: Path):
    """Extract target and selected direction from a session log file."""
    target = selected = None
    with open(session_path, "r") as f:
        for line in f:
            if "Target Direction:" in line:
                target = line.split(":")[-1].strip().lower()
            elif "Selected Direction:" in line:
                selected = line.split(":")[-1].strip().lower()
    return target, selected


def discover_test_files(data_dir: Path):
    eeg_dir = data_dir / "eeg"
    sessions_dir = data_dir / "sessions"

    eeg_files = sorted(eeg_dir.glob("eeg_recording_*.h5"))
    session_files = sorted(sessions_dir.glob("session_*.txt"))
    trigger_files = sorted(sessions_dir.glob("triggers_*.txt"))

    print(f"Found {len(eeg_files)} EEG recordings")
    print(f"Found {len(session_files)} session files")
    print(f"Found {len(trigger_files)} trigger files")

    eeg_records = []
    for p in eeg_files:
        ts = parse_eeg_timestamp(p.name)
        if ts:
            eeg_records.append({"path": p, "timestamp": ts})
    eeg_records.sort(key=lambda x: x["timestamp"])

    session_by_ts = {}
    for p in session_files:
        ts = parse_txt_timestamp(p.name)
        if ts:
            session_by_ts[ts] = p

    trigger_by_ts = {}
    for p in trigger_files:
        ts = parse_txt_timestamp(p.name)
        if ts:
            trigger_by_ts[ts] = p

    common_ts = sorted(set(session_by_ts) & set(trigger_by_ts))

    matched = []
    for sess_ts in common_ts:
        best_eeg = None
        for rec in eeg_records:
            if rec["timestamp"] <= sess_ts:
                best_eeg = rec
        if best_eeg is None:
            print(f"  WARNING: No EEG recording for session {sess_ts}")
            continue
        matched.append({
            "hdf5_path": best_eeg["path"],
            "session_path": session_by_ts[sess_ts],
            "trigger_path": trigger_by_ts[sess_ts],
            "timestamp": sess_ts,
        })

    print(f"\nMatched {len(matched)} runs")
    return matched


def run_preprocessing(
    test_data_dir: Path = DEFAULT_TEST_DATA_DIR,
    output_dir: Path = DEFAULT_TEST_OUTPUT_DIR,
) -> Path:
    """Run preprocessing and return the path to the saved npz file."""
    test_data_dir = Path(test_data_dir)
    output_dir = Path(output_dir)

    print("Discovering test files...\n")
    matched_runs = discover_test_files(test_data_dir)

    if not matched_runs:
        raise RuntimeError(
            f"No matched runs found under {test_data_dir}. "
            "Expected eeg/*.h5 and sessions/{session,triggers}_*.txt"
        )

    recordings = {}
    for run in matched_runs:
        key = str(run["hdf5_path"])
        recordings.setdefault(key, []).append(run)

    print(
        f"\nProcessing {len(matched_runs)} sessions "
        f"across {len(recordings)} recordings...\n"
    )

    all_epochs = []
    all_events = []
    session_results = []
    run_index = 0

    for eeg_path, runs in recordings.items():
        eeg_name = Path(eeg_path).name
        print(f"Recording: {eeg_name}")

        eeg_raw, rec_start_unix = load_test_eeg(Path(eeg_path))
        eeg_filtered = apply_bandpass_filter(eeg_raw)
        bad_channels, _ = detect_bad_channels(eeg_filtered)
        eeg_car = apply_car(eeg_filtered, bad_channels)

        bad_str = str(bad_channels) if bad_channels else "none"
        print(
            f"  {eeg_raw.shape[0]} samples ({eeg_raw.shape[0] / SR:.1f}s), "
            f"bad_ch={bad_str}"
        )

        for run in runs:
            events = parse_trigger_file(run["trigger_path"])
            offset_ms = compute_recording_offset(
                run["session_path"], rec_start_unix
            )
            epochs, valid_events = extract_epochs(eeg_car, events, offset_ms)
            target_dir, selected_dir = parse_session_result(run["session_path"])

            for ev in valid_events:
                ev["run_index"] = run_index

            n_t = sum(1 for e in valid_events if e["is_target"])
            n_nt = len(valid_events) - n_t
            print(
                f"  Run {run_index:>2d}: {Path(run['session_path']).name} "
                f"\u2014 {len(valid_events)} epochs ({n_t}T/{n_nt}NT), "
                f"target={target_dir}, online={selected_dir}"
            )

            all_epochs.append(epochs)
            all_events.extend(valid_events)
            session_results.append({
                "run_index": run_index,
                "target": target_dir,
                "selected": selected_dir,
            })
            run_index += 1

    if not all_epochs:
        raise RuntimeError("No sessions processed successfully.")

    combined = np.concatenate(all_epochs, axis=0)
    labels = np.array(
        [1 if e["is_target"] else 0 for e in all_events], dtype=np.int8
    )
    directions = np.array(
        [DIRECTION_MAP[e["direction"]] for e in all_events], dtype=np.int8
    )
    targets = np.array(
        [DIRECTION_MAP[e["target"]] for e in all_events], dtype=np.int8
    )
    run_indices = np.array(
        [e["run_index"] for e in all_events], dtype=np.int16
    )
    day_labels = np.ones_like(run_indices)

    print("\nApplying baseline correction...")
    epochs_bc = apply_baseline_correction(combined)

    is_clean, rejection_reasons = reject_artifacts(epochs_bc)
    n_total = len(is_clean)
    n_clean = int(is_clean.sum())
    print(f"  Clean: {n_clean}/{n_total} ({n_clean / n_total * 100:.1f}%)")

    online_targets = np.array(
        [
            DIRECTION_MAP[r["target"]] if r["target"] in DIRECTION_MAP else -1
            for r in session_results
        ],
        dtype=np.int8,
    )
    online_selected = np.array(
        [
            DIRECTION_MAP[r["selected"]] if r["selected"] in DIRECTION_MAP else -1
            for r in session_results
        ],
        dtype=np.int8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "clean_epochs.npz"
    np.savez_compressed(
        out_path,
        epochs=epochs_bc,
        labels=labels,
        directions=directions,
        targets=targets,
        run_indices=run_indices,
        day_labels=day_labels,
        is_clean=is_clean,
        rejection_reasons=np.array(rejection_reasons),
        sr=np.array(SR),
        epoch_pre_ms=np.array(EPOCH_PRE_MS),
        epoch_post_ms=np.array(EPOCH_POST_MS),
        online_targets=online_targets,
        online_selected=online_selected,
    )

    mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out_path} ({mb:.1f} MB)")
    print(f"  Shape: {epochs_bc.shape}")
    print(f"  Sessions: {len(session_results)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=DEFAULT_TEST_DATA_DIR,
        help=f"Directory containing eeg/ and sessions/ (default: {DEFAULT_TEST_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TEST_OUTPUT_DIR,
        help=f"Output directory for clean_epochs.npz (default: {DEFAULT_TEST_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    try:
        run_preprocessing(args.test_data_dir, args.output_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
