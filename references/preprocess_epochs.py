import csv
import sys
from pathlib import Path

import numpy as np

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    SR,
    EPOCH_PRE_MS,
    EPOCH_POST_MS,
    DIRECTION_MAP,
    BASELINE_START_MS,
    BASELINE_END_MS,
)
from utils import (
    discover_and_match_files,
    load_eeg_from_hdf5,
    parse_trigger_file,
    compute_recording_offset,
    extract_epochs,
    apply_bandpass_filter,
    detect_bad_channels,
    apply_car,
    apply_baseline_correction,
    reject_artifacts,
)


def process_single_run(run_info, run_index):
    hdf5_name = Path(run_info["hdf5_path"]).name
    print(f"  Run {run_index:>2d}: {hdf5_name}")

    eeg_raw, recording_start_unix = load_eeg_from_hdf5(run_info["hdf5_path"])
    events = parse_trigger_file(run_info["trigger_path"])
    offset_ms = compute_recording_offset(
        run_info["session_path"], recording_start_unix
    )

    n_samples = eeg_raw.shape[0]
    duration_s = n_samples / SR
    n_target = sum(1 for e in events if e["is_target"])
    n_nontarget = len(events) - n_target

    # Bandpass filter (0.5 - 30 Hz)
    eeg_filtered = apply_bandpass_filter(eeg_raw)

    # Detect bad channels and apply CAR excluding those channels
    bad_channels, channel_stds = detect_bad_channels(eeg_filtered)
    eeg_car = apply_car(eeg_filtered, bad_channels)

    # Epoching
    epochs, valid_events = extract_epochs(eeg_car, events, offset_ms)

    for event in valid_events:
        event["run_index"] = run_index

    bad_ch_str = str(bad_channels) if bad_channels else "none"
    print(
        f"           {n_samples} samples ({duration_s:.1f}s), "
        f"{len(events)} events ({n_target}T/{n_nontarget}NT), "
        f"bad_ch={bad_ch_str}"
    )

    run_stats = {
        "run_index": run_index,
        "hdf5_file": hdf5_name,
        "n_samples": n_samples,
        "duration_s": round(duration_s, 1),
        "n_events_total": len(events),
        "n_target": n_target,
        "n_nontarget": n_nontarget,
        "n_epochs_extracted": len(valid_events),
        "offset_ms": round(offset_ms, 1),
        "bad_channels": bad_channels,
        "channel_stds": channel_stds.tolist(),
        "session_date": run_info["timestamp"].date().isoformat(),
    }

    return epochs, valid_events, run_stats


def build_epoch_arrays(all_epochs, all_events, all_run_stats):
    combined_epochs = np.concatenate(all_epochs, axis=0)

    labels = np.array(
        [1 if e["is_target"] else 0 for e in all_events],
        dtype=np.int8
    )
    directions = np.array(
        [DIRECTION_MAP[e["direction"]] for e in all_events],
        dtype=np.int8
    )
    targets = np.array(
        [DIRECTION_MAP[e["target"]] for e in all_events],
        dtype=np.int8
    )
    run_indices = np.array(
        [e["run_index"] for e in all_events],
        dtype=np.int16
    )

    run_ids_processed = np.array(
        [s["run_index"] for s in all_run_stats],
        dtype=np.int16
    )

    max_bad = max((len(s["bad_channels"]) for s in all_run_stats), default=0)
    if max_bad == 0:
        bad_ch_array = np.array([], dtype=np.int8).reshape(len(all_run_stats), 0)
    else:
        bad_ch_array = np.full((len(all_run_stats), max_bad), -1, dtype=np.int8)
        for row_idx, stats in enumerate(all_run_stats):
            for j, ch in enumerate(stats["bad_channels"]):
                bad_ch_array[row_idx, j] = ch

    return (
        combined_epochs,
        labels,
        directions,
        targets,
        run_indices,
        run_ids_processed,
        bad_ch_array,
    )


def build_day_labels(run_indices, all_run_stats):
    run_to_day = {}
    for stats in all_run_stats:
        run_to_day[stats["run_index"]] = stats["run_index"] // 10 + 1

    day_labels = np.array([run_to_day[int(r)] for r in run_indices], dtype=np.int16)
    return day_labels


def save_epoch_metadata_csv(all_events, is_clean, rejection_reasons, output_path):
    csv_path = output_path / "epoch_metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch_idx",
            "run_index",
            "trigger_time_ms",
            "direction",
            "target_direction",
            "is_target",
            "is_clean",
            "rejection_reason",
        ])
        for i, event in enumerate(all_events):
            writer.writerow([
                i,
                event["run_index"],
                f"{event['time_ms']:.3f}",
                event["direction"],
                event["target"],
                int(event["is_target"]),
                int(is_clean[i]),
                rejection_reasons[i],
            ])
    print(f"Saved: {csv_path}")


def main():
    print("Discovering files...\n")
    matched_runs = discover_and_match_files(DATA_DIR)

    if not matched_runs:
        print("\nERROR: No matched runs found.")
        print(f"  Searched in: {DATA_DIR}")
        print("  Make sure DATA_DIR in config.py points to your data folder.")
        sys.exit(1)

    print(f"\nProcessing {len(matched_runs)} runs...\n")

    all_epochs = []
    all_events = []
    all_run_stats = []
    failed_runs = []

    for i, run_info in enumerate(matched_runs):
        try:
            epochs, events, stats = process_single_run(run_info, i)
            all_epochs.append(epochs)
            all_events.extend(events)
            all_run_stats.append(stats)
        except Exception as e:
            print(f"  ERROR on run {i}: {e}")
            print("  Skipping this run.\n")
            failed_runs.append((i, str(e)))

    if not all_epochs:
        print("\nERROR: No runs were successfully processed.")
        sys.exit(1)

    (
        combined_epochs,
        labels,
        directions,
        targets,
        run_indices,
        run_ids_processed,
        bad_ch_array,
    ) = build_epoch_arrays(all_epochs, all_events, all_run_stats)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nApplying baseline correction (-100 to 0 ms)...\n")
    epochs_baseline = apply_baseline_correction(combined_epochs)

    pre_samp = int(EPOCH_PRE_MS * SR / 1000)
    bl_start = pre_samp + int(BASELINE_START_MS * SR / 1000)
    bl_end = pre_samp + int(BASELINE_END_MS * SR / 1000)
    bl_mean = epochs_baseline[:, bl_start:bl_end, :].mean()
    print(f"  Mean in baseline window after correction: {bl_mean:.6f} µV")

    print("\nRunning artifact rejection and saving clean output...\n")
    is_clean, rejection_reasons = reject_artifacts(epochs_baseline)
    day_labels = build_day_labels(run_indices, all_run_stats)

    n_total = len(is_clean)
    n_clean = int(is_clean.sum())
    n_rejected = n_total - n_clean
    clean_target = int((is_clean & (labels == 1)).sum())
    clean_nontarget = int((is_clean & (labels == 0)).sum())

    clean_path = output_path / "clean_epochs.npz"
    np.savez_compressed(
        clean_path,
        epochs=epochs_baseline,
        labels=labels,
        directions=directions,
        targets=targets,
        run_indices=run_indices,
        day_labels=day_labels,
        is_clean=is_clean,
        rejection_reasons=np.array(rejection_reasons),
        bad_channels_per_run=bad_ch_array,
        bad_channel_run_indices=run_ids_processed,
        sr=np.array(SR),
        epoch_pre_ms=np.array(EPOCH_PRE_MS),
        epoch_post_ms=np.array(EPOCH_POST_MS),
    )
    clean_file_mb = clean_path.stat().st_size / 1024 / 1024
    print(f"Saved: {clean_path} ({clean_file_mb:.1f} MB)")
    print(f"  Epochs shape: {epochs_baseline.shape}")
    print(f"  Clean epochs: {n_clean} ({n_clean / n_total * 100:.1f}%)")
    print(f"  Rejected:     {n_rejected} ({n_rejected / n_total * 100:.1f}%)")
    print(f"  Clean target:     {clean_target} / {(labels == 1).sum()}")
    print(f"  Clean non-target: {clean_nontarget} / {(labels == 0).sum()}")

    save_epoch_metadata_csv(all_events, is_clean, rejection_reasons, output_path)


if __name__ == "__main__":
    main()
