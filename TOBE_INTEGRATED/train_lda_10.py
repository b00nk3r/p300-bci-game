"""
Train a single-trial LDA on calibration epochs.

Mirrors the logic of ``TOBE_INTEGRATED/SingleTrialLDA_10.ipynb`` but as
a standalone script so the live game pipeline can call it after
preprocessing completes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PREP_DIR = PROJECT_ROOT / "model preprocessing"
if str(MODEL_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_PREP_DIR))

from utils import extract_single_trial_features  # noqa: E402


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_processed" / "clean_epochs.npz"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "10trials_model.joblib"

N_RUNS_DEFAULT = 10
WINDOW_START = 0
WINDOW_END = 800
DEC_WINDOW = 20
DEC_STEP = 10


def train_model(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    n_runs: int = N_RUNS_DEFAULT,
) -> Path:
    """Train an LDA on the preprocessed calibration epochs and save it."""
    data_path = Path(data_path)
    model_path = Path(model_path)

    data = np.load(data_path)
    epochs = data["epochs"]
    labels = data["labels"]
    directions = data["directions"]
    targets = data["targets"]
    run_indices = data["run_indices"]
    day_labels = data["day_labels"]
    is_clean = data["is_clean"]

    run_mask = run_indices < n_runs
    epochs = epochs[run_mask]
    labels = labels[run_mask]
    directions = directions[run_mask]
    targets = targets[run_mask]
    run_indices = run_indices[run_mask]
    day_labels = day_labels[run_mask]
    is_clean = is_clean[run_mask]

    n_total = len(is_clean)
    print(f"Loaded: {n_total} epochs")
    print(
        f"Params: window={WINDOW_START}-{WINDOW_END}ms, "
        f"dec={DEC_WINDOW}/{DEC_STEP}"
    )

    X, y, days, trial_ids, epoch_dirs, trial_targets = extract_single_trial_features(
        epochs, labels, directions, targets,
        run_indices, day_labels, is_clean,
        window_start_ms=WINDOW_START,
        window_end_ms=WINDOW_END,
        dec_window=DEC_WINDOW,
        dec_step=DEC_STEP,
    )

    if X.shape[0] == 0:
        raise RuntimeError(
            "No clean epochs found after feature extraction. "
            "Check that calibration data was recorded correctly."
        )

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(X, y)

    artifact = {
        "model_name": "single_trial_lda_calibration",
        "model": model,
        "model_params": {"solver": "lsqr", "shrinkage": "auto"},
        "feature_params": {
            "window_start_ms": WINDOW_START,
            "window_end_ms": WINDOW_END,
            "dec_window": DEC_WINDOW,
            "dec_step": DEC_STEP,
        },
        "feature_normalization": "none",
        "n_train_runs": n_runs,
        "n_train_epochs": int(len(y)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    print(
        f"Trained on {len(np.unique(trial_ids))} runs, "
        f"{X.shape[0]} epochs, {X.shape[1]} features"
    )
    print(f"Saved to: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-runs", type=int, default=N_RUNS_DEFAULT)
    args = parser.parse_args()

    try:
        train_model(args.data_path, args.model_path, args.n_runs)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
