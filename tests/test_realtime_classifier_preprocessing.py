import importlib
import sys
import types

import numpy as np
from scipy.signal import butter, filtfilt


def _load_realtime_classifier(monkeypatch):
    fake_pylsl = types.SimpleNamespace(
        StreamInlet=object,
        resolve_byprop=lambda *args, **kwargs: [],
        local_clock=lambda: 0.0,
    )
    monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)

    import realtime_classifier as rc

    return importlib.reload(rc)


def _make_classifier(rc_module):
    clf = rc_module.RealtimeClassifier.__new__(rc_module.RealtimeClassifier)
    clf.b_30hz, clf.a_30hz = butter(
        rc_module.BPF_ORDER,
        [rc_module.BPF_LOW, rc_module.BPF_HIGH],
        btype="bandpass",
        fs=rc_module.SR,
    )
    return clf


def _build_continuous_data(rc_module, n_samples=2500):
    t = np.arange(n_samples, dtype=np.float64) / rc_module.SR
    data = np.zeros((n_samples, rc_module.N_CHANNELS), dtype=np.float64)

    data[:, 0] = 0.0
    for ch in range(1, rc_module.N_CHANNELS):
        data[:, ch] = (
            8.0 * np.sin(2 * np.pi * (0.6 + 0.05 * ch) * t + (0.1 * ch))
            + 1.5 * np.cos(2 * np.pi * 8.0 * t + (0.2 * ch))
            + (0.05 * ch * t)
        )

    return data, t


def _manual_preprocess(rc_module, clf, data, event_sample_indices, directions):
    eeg_filtered = filtfilt(clf.b_30hz, clf.a_30hz, data, axis=0)
    bad_channels = clf._detect_bad_channels(eeg_filtered)
    eeg_car = clf._apply_car(eeg_filtered, bad_channels)

    pre_samples = int(rc_module.EPOCH_PRE_MS * rc_module.SR / 1000)
    post_samples = int(rc_module.EPOCH_POST_MS * rc_module.SR / 1000)
    raw_epochs = np.array(
        [eeg_car[idx - pre_samples:idx + post_samples] for idx in event_sample_indices],
        dtype=np.float64,
    )
    epochs_corrected = clf._apply_baseline_correction(raw_epochs)
    is_clean = clf._artifact_mask(epochs_corrected)

    return epochs_corrected, is_clean, np.array(directions, dtype=np.int8), bad_channels


def test_preprocess_trial_matches_offline_order(monkeypatch):
    rc = _load_realtime_classifier(monkeypatch)
    clf = _make_classifier(rc)
    data, timestamps = _build_continuous_data(rc)

    event_indices = [450, 1050, 1650]
    directions = ["up", "left", "down"]
    events = [
        {"timestamp": float(timestamps[idx]), "direction": direction}
        for idx, direction in zip(event_indices, directions)
    ]

    actual_epochs, actual_clean, actual_dirs = clf._preprocess_trial(
        data, timestamps, events
    )
    expected_epochs, expected_clean, expected_dirs, bad_channels = _manual_preprocess(
        rc,
        clf,
        data,
        event_indices,
        [rc.DIRECTION_MAP[d] for d in directions],
    )

    assert 0 in bad_channels
    np.testing.assert_allclose(actual_epochs, expected_epochs)
    np.testing.assert_array_equal(actual_clean, expected_clean)
    np.testing.assert_array_equal(actual_dirs, expected_dirs)


def test_preprocess_trial_drops_events_without_full_epoch(monkeypatch):
    rc = _load_realtime_classifier(monkeypatch)
    clf = _make_classifier(rc)
    data, timestamps = _build_continuous_data(rc, n_samples=1800)

    events = [
        {"timestamp": float(timestamps[50]), "direction": "up"},
        {"timestamp": float(timestamps[900]), "direction": "right"},
    ]

    epochs_corrected, is_clean, directions = clf._preprocess_trial(
        data, timestamps, events
    )

    assert epochs_corrected.shape[0] == 1
    np.testing.assert_array_equal(is_clean, np.array([True]))
    np.testing.assert_array_equal(
        directions,
        np.array([rc.DIRECTION_MAP["right"]], dtype=np.int8),
    )


