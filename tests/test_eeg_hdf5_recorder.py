import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from src.data.eeg_hdf5_recorder import EEGHDF5Recorder


def test_eeg_hdf5_recorder_writes_samples_timestamps_and_metadata(tmp_path):
    """Recorder should persist continuous EEG chunks with timing metadata."""
    recorder = EEGHDF5Recorder(output_dir=str(tmp_path), flush_interval_s=0.0)

    filepath = recorder.start(
        sampling_rate=500,
        n_channels=2,
        stream_name="TestStream",
        stream_type="EEG",
        device_name="TestDevice",
        scan_count=4,
        bandpass_low_hz=0.1,
        bandpass_high_hz=100.0,
        bandpass_order=4,
        notch_low_hz=48.0,
        notch_high_hz=52.0,
        notch_order=4,
        session_start_unix_s=1000.5,
        session_start_unix_ns=1000500000000,
        session_start_lsl_s=50.25,
        unix_lsl_offset_s=950.25,
        timestamp_reference="test reference",
    )

    samples = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    lsl_timestamps = np.array([10.0, 10.002, 10.004], dtype=np.float64)
    unix_timestamps = lsl_timestamps + 950.25

    recorder.append_chunk(samples, lsl_timestamps, unix_timestamps)
    recorder.close(recording_end_unix_s=1001.0, recording_end_unix_ns=1001000000000)

    with h5py.File(filepath, "r") as handle:
        np.testing.assert_allclose(handle["samples"][:], samples)
        np.testing.assert_allclose(handle["lsl_timestamps"][:], lsl_timestamps)
        np.testing.assert_allclose(handle["unix_timestamps"][:], unix_timestamps)
        np.testing.assert_array_equal(handle["sample_index"][:], np.array([0, 1, 2], dtype=np.int64))

        assert handle.attrs["stream_name"] == "TestStream"
        assert handle.attrs["device_name"] == "TestDevice"
        assert handle.attrs["session_start_unix_s"] == 1000.5
        assert handle.attrs["session_start_lsl_s"] == 50.25
        assert handle.attrs["n_samples_written"] == 3
        assert handle.attrs["duration_s"] == 0.5
