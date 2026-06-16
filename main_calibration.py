import sys

from src.app.application import run_main
from src.app.calibration_mode import CalibrationMode


if __name__ == "__main__":
    sys.exit(run_main(CalibrationMode, default_width=3072, default_height=1920))