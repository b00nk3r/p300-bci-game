from pathlib import Path

from config import Direction
from src.data.session_logger import SessionLogger


def test_session_logger_records_target_and_selected_direction(tmp_path):
    """Saved trial logs should include both the intended target and model output."""
    logger = SessionLogger(output_dir=str(tmp_path))

    logger.start_session(
        flash_duration_ms=100,
        isi_ms=125,
        num_sequences=10,
        inter_sequence_pause_ms=0,
        flash_pattern="RANDOM",
        color_scheme="GRAY_WHITE",
        target_direction=Direction.LEFT,
    )
    logger.log_flash_start(Direction.UP, sequence=0, timestamp_ms=0.0)
    logger.log_flash_end(Direction.UP, sequence=0, timestamp_ms=100.0)

    filepath = logger.end_session(selected_direction=Direction.DOWN)
    contents = Path(filepath).read_text()

    assert "Target Direction:    left" in contents
    assert "Selected Direction:  down" in contents
