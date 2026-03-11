import numpy as np
import h5py
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from scipy.signal import butter, filtfilt
from config import (
    SR, EPOCH_PRE_MS, EPOCH_POST_MS,
    BPF_LOW, BPF_HIGH, BPF_ORDER,
    BAD_CH_LOW_FACTOR, BAD_CH_HIGH_FACTOR,
    BASELINE_START_MS, BASELINE_END_MS, 
    ARTIFACT_PP_THRESHOLD,
)


def parse_timestamp_from_hdf5_name(filename):
    match = re.search(
        r'Game(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})', filename
    )
    if match:
        return datetime(
            int(match.group(1)), int(match.group(2)), int(match.group(3)),
            int(match.group(4)), int(match.group(5)), int(match.group(6))
        )
    return None


def parse_timestamp_from_txt_name(filename):
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        d = match.group(1)
        t = match.group(2)
        return datetime(
            int(d[0:4]), int(d[4:6]), int(d[6:8]),
            int(t[0:2]), int(t[2:4]), int(t[4:6])
        )
    return None


def discover_and_match_files(data_dir):
    eeg_data_path = Path(data_dir) / "eeg"
    log_data_path = Path(data_dir) / "sessions_log"

    hdf5_files = sorted(eeg_data_path.rglob("Game*.hdf5"))
    session_files = sorted(log_data_path.rglob("session_*.txt"))
    trigger_files = sorted(log_data_path.rglob("triggers_*.txt"))

    print(f"Found {len(hdf5_files)} HDF5 files")
    print(f"Found {len(session_files)} session files")
    print(f"Found {len(trigger_files)} trigger files")

    hdf5_by_ts = {}
    for path in hdf5_files:
        ts = parse_timestamp_from_hdf5_name(path.name)
        if ts:
            hdf5_by_ts[ts] = path

    session_by_ts = {}
    for path in session_files:
        ts = parse_timestamp_from_txt_name(path.name)
        if ts:
            session_by_ts[ts] = path

    trigger_by_ts = {}
    for path in trigger_files:
        ts = parse_timestamp_from_txt_name(path.name)
        if ts:
            trigger_by_ts[ts] = path

    common_timestamps = sorted(
        set(session_by_ts.keys()) & set(trigger_by_ts.keys())
    )

    matched_runs = []
    unmatched = []

    for session_ts in common_timestamps:
        best_hdf5 = None
        best_diff = float("inf")

        for hdf5_ts, hdf5_path in hdf5_by_ts.items():
            diff = (session_ts - hdf5_ts).total_seconds()
            if 0 < diff < 60 and diff < best_diff:
                best_diff = diff
                best_hdf5 = hdf5_path

        if best_hdf5 is None:
            unmatched.append(session_ts)
            continue

        matched_runs.append({
            "hdf5_path": best_hdf5,
            "session_path": session_by_ts[session_ts],
            "trigger_path": trigger_by_ts[session_ts],
            "timestamp": session_ts,
        })

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} session(s) had no matching HDF5:")
        for ts in unmatched:
            print(f"  {ts}")

    print(f"\nSuccessfully matched {len(matched_runs)} run triplets")
    return matched_runs


def load_eeg_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        eeg = f["RawData/Samples"][:]

        desc_xml = f["RawData/AcquisitionTaskDescription"][0].decode("utf-8")
        root = ET.fromstring(desc_xml)
        rec_start_str = root.find(".//RecordingDateBegin").text

        rec_start_dt = datetime.fromisoformat(
            rec_start_str.replace("Z", "+00:00")
        )
        recording_start_unix = rec_start_dt.timestamp()

    return eeg, recording_start_unix


def parse_trigger_file(trigger_path):
    events = []

    with open(trigger_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or not line[0].isdigit():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue

            time_ms = float(parts[0])
            label = parts[1]
            target = parts[2]

            if not label.startswith("flash_"):
                continue

            direction = label.replace("flash_", "")

            events.append({
                "time_ms": time_ms,
                "direction": direction.lower(),
                "target": target.lower(),
                "is_target": direction.lower() == target.lower(),
            })

    return events


def compute_recording_offset(session_path, recording_start_unix):
    with open(session_path, "r") as f:
        for line in f:
            parts = line.split()

            if len(parts) >= 5 and parts[4] == "flash_start":
                trigger_ms = float(parts[0])
                absolute_time = float(parts[1])

                session_start_unix = absolute_time - trigger_ms / 1000.0

                offset_ms = (session_start_unix - recording_start_unix) * 1000.0
                return offset_ms

    raise ValueError(
        f"Could not find absolute timestamps in {session_path}. "
        f"Make sure the file contains 'flash_start' events with absolute times."
    )


def extract_epochs(eeg, events, offset_ms, sr=SR,
                   pre_ms=EPOCH_PRE_MS, post_ms=EPOCH_POST_MS):
    pre_samples = int(pre_ms * sr / 1000)
    post_samples = int(post_ms * sr / 1000)

    epochs = []
    valid_events = []

    for event in events:
        sample_idx = int((offset_ms + event["time_ms"]) * sr / 1000)

        start = sample_idx - pre_samples
        end = sample_idx + post_samples

        if start < 0 or end > eeg.shape[0]:
            continue

        epoch = eeg[start:end, :]
        epochs.append(epoch)
        valid_events.append(event)

    epochs_array = np.array(epochs, dtype=np.float32)

    return epochs_array, valid_events


def apply_bandpass_filter(eeg, low=BPF_LOW, high=BPF_HIGH, sr=SR, order=BPF_ORDER):
    b, a = butter(order, [low, high], btype="bandpass", fs=sr)
    filtered = filtfilt(b, a, eeg, axis=0)
    return filtered


def detect_bad_channels(eeg_filtered):
    channel_stds = eeg_filtered.std(axis=0)
    median_std = np.median(channel_stds)

    bad_channels = []
    for ch in range(eeg_filtered.shape[1]):
        if channel_stds[ch] < BAD_CH_LOW_FACTOR * median_std:
            bad_channels.append(ch)
        elif channel_stds[ch] > BAD_CH_HIGH_FACTOR * median_std:
            bad_channels.append(ch)

    return bad_channels, channel_stds


def apply_car(eeg, bad_channels):
    good_channels = [
        ch for ch in range(eeg.shape[1])
        if ch not in bad_channels
    ]

    avg = eeg[:, good_channels].mean(axis=1, keepdims=True)
    eeg_car = eeg - avg

    return eeg_car


def apply_baseline_correction(epochs, sr=SR, baseline_start_ms=BASELINE_START_MS, baseline_end_ms=BASELINE_END_MS, pre_ms=EPOCH_PRE_MS):
    pre_samples = int(pre_ms * sr / 1000)

    bl_start = pre_samples + int(baseline_start_ms * sr / 1000)
    bl_end = pre_samples + int(baseline_end_ms * sr / 1000)

    baseline_mean = epochs[:, bl_start:bl_end, :].mean(axis=1, keepdims=True)

    return epochs - baseline_mean


def reject_artifacts(epochs, pp_threshold=ARTIFACT_PP_THRESHOLD):
    n_epochs = epochs.shape[0]
    is_clean = np.ones(n_epochs, dtype=bool)
    rejection_reasons = [""] * n_epochs

    for i in range(n_epochs):
        ep = epochs[i]  # (epoch_length, n_channels)

        pp = ep.max(axis=0) - ep.min(axis=0)
        max_pp = pp.max()
        if max_pp > pp_threshold:
            is_clean[i] = False
            worst_ch = pp.argmax()
            rejection_reasons[i] = f"pp_ch{worst_ch}_{max_pp:.1f}uV"

    return is_clean, rejection_reasons


def overlapping_moving_average(data, window_size, step_size):
    n_time, n_channels = data.shape
    n_output = (n_time - window_size) // step_size + 1

    decimated = np.zeros((n_output, n_channels), dtype=data.dtype)
    for i in range(n_output):
        start = i * step_size
        end = start + window_size
        decimated[i] = data[start:end, :].mean(axis=0)

    return decimated

def identify_trials(targets_run):
    trials = []
    current_target = targets_run[0]
    current_start = 0

    for i in range(1, len(targets_run)):
        if targets_run[i] != current_target:
            trials.append({
                "target_dir": int(current_target),
                "epoch_indices": np.arange(current_start, i),
            })
            current_target = targets_run[i]
            current_start = i

    # Last trial
    trials.append({
        "target_dir": int(current_target),
        "epoch_indices": np.arange(current_start, len(targets_run)),
    })

    return trials


def extract_averaged_features(epochs, labels, directions, targets, run_indices, day_labels, is_clean, window_start_ms, window_end_ms, dec_window, dec_step, sr=SR, pre_ms=EPOCH_PRE_MS):
    pre_samp = int(pre_ms * sr / 1000)
    start_idx = pre_samp + int(window_start_ms * sr / 1000)
    end_idx = pre_samp + int(window_end_ms * sr / 1000)

    X_list = []
    y_list = []
    day_list = []
    trial_id_list = []

    trial_counter = 0

    for run in np.unique(run_indices):
        run_mask = run_indices == run
        run_global_indices = np.where(run_mask)[0]

        day = day_labels[run_mask][0]
        run_targets = targets[run_mask]

        trials = identify_trials(run_targets)

        for trial_info in trials:
            target_dir = trial_info["target_dir"]
            trial_global = run_global_indices[trial_info["epoch_indices"]]

            for direction in range(4):
                dir_mask = directions[trial_global] == direction
                clean_mask = is_clean[trial_global]
                selected_global = trial_global[dir_mask & clean_mask]

                if len(selected_global) == 0:
                    continue

                selected_epochs = epochs[selected_global]

                averaged = selected_epochs.mean(axis=0)
                trimmed = averaged[start_idx:end_idx, :]
                decimated = overlapping_moving_average(
                    trimmed, dec_window, dec_step
                )
                flat = decimated.flatten()
                X_list.append(flat)
                y_list.append(1 if direction == target_dir else 0)
                day_list.append(day)
                trial_id_list.append(trial_counter)

            trial_counter += 1

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int8)
    days = np.array(day_list)
    trial_ids = np.array(trial_id_list)

    return X, y, days, trial_ids
    

def compute_trial_accuracy(scores, labels, run_ids):
    per_run = []

    for run in np.unique(run_ids):
        run_mask = run_ids == run
        run_scores = scores[run_mask]
        run_labels = labels[run_mask]

        predicted_idx = np.argmax(run_scores)
        target_idx = np.argmax(run_labels)
        correct = predicted_idx == target_idx
        per_run.append({
            "run_id": int(run),
            "correct": bool(correct),
            "predicted_idx": int(predicted_idx),
            "target_idx": int(target_idx),
        })

    accuracy = sum(r["correct"] for r in per_run) / len(per_run)
    return accuracy, per_run


def compute_per_day_accuracy(per_run_details, days_feat, run_ids_feat):
    run_to_day = {}
    for run in np.unique(run_ids_feat):
        mask = run_ids_feat == run
        run_to_day[int(run)] = int(days_feat[mask][0])

    # Group trial results by day
    day_results = {}
    for trial in per_run_details:
        day = run_to_day[trial["run_id"]]
        if day not in day_results:
            day_results[day] = []
        day_results[day].append(trial["correct"])

    day_accuracies = {}
    for day in sorted(day_results.keys()):
        results = day_results[day]
        day_accuracies[day] = sum(results) / len(results)

    return day_accuracies


def extract_single_trial_features(epochs, labels, directions, targets, run_indices, day_labels, is_clean, window_start_ms, window_end_ms, dec_window, dec_step, sr=SR, pre_ms=EPOCH_PRE_MS):
    pre_samp = int(pre_ms * sr / 1000)
    start_idx = pre_samp + int(window_start_ms * sr / 1000)
    end_idx = pre_samp + int(window_end_ms * sr / 1000)

    X_list = []
    y_list = []
    day_list = []
    trial_id_list = []
    direction_list = []
    target_dir_list = []

    trial_counter = 0

    for run in np.unique(run_indices):
        run_mask = run_indices == run
        run_global_indices = np.where(run_mask)[0]

        day = day_labels[run_mask][0]
        run_targets = targets[run_mask]

        trials = identify_trials(run_targets)

        for trial_info in trials:
            target_dir = trial_info["target_dir"]
            trial_global = run_global_indices[trial_info["epoch_indices"]]
            for idx in trial_global:
                if not is_clean[idx]:
                    continue

                epoch = epochs[idx]
                trimmed = epoch[start_idx:end_idx, :]
                decimated = overlapping_moving_average(
                    trimmed, dec_window, dec_step
                )
                flat = decimated.flatten()

                direction = int(directions[idx])

                X_list.append(flat)
                y_list.append(1 if direction == target_dir else 0)
                day_list.append(day)
                trial_id_list.append(trial_counter)
                direction_list.append(direction)
                target_dir_list.append(target_dir)

            trial_counter += 1

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int8)
    days = np.array(day_list)
    trial_ids = np.array(trial_id_list)
    epoch_directions = np.array(direction_list)
    trial_targets = np.array(target_dir_list)

    return X, y, days, trial_ids, epoch_directions, trial_targets


def compute_trial_accuracy_from_single_trials(scores, trial_ids, epoch_directions, trial_targets):
    per_trial = []

    for trial_id in np.unique(trial_ids):
        trial_mask = trial_ids == trial_id
        trial_scores = scores[trial_mask]
        trial_dirs = epoch_directions[trial_mask]
        true_target = trial_targets[trial_mask][0]

        # Average score per direction
        dir_scores = {}
        for d in range(4):
            d_mask = trial_dirs == d
            if d_mask.sum() > 0:
                dir_scores[d] = trial_scores[d_mask].mean()
            else:
                dir_scores[d] = -np.inf

        predicted_dir = max(dir_scores, key=dir_scores.get)
        correct = predicted_dir == true_target
        per_trial.append({
            "trial_id": int(trial_id),
            "correct": bool(correct),
            "predicted_dir": int(predicted_dir),
            "true_target": int(true_target),
            "dir_scores": dir_scores,
        })

    accuracy = sum(t["correct"] for t in per_trial) / len(per_trial)
    return accuracy, per_trial


def compute_per_day_accuracy_single_trial(per_trial_details, days, trial_ids):
    trial_to_day = {}
    for trial_id in np.unique(trial_ids):
        mask = trial_ids == trial_id
        trial_to_day[int(trial_id)] = int(days[mask][0])

    day_results = {}
    for trial in per_trial_details:
        day = trial_to_day[trial["trial_id"]]
        if day not in day_results:
            day_results[day] = []
        day_results[day].append(trial["correct"])

    day_accuracies = {}
    for day in sorted(day_results.keys()):
        results = day_results[day]
        day_accuracies[day] = sum(results) / len(results)

    return day_accuracies