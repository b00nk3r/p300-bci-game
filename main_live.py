import sys

from src.app.application import run_main
from src.app.live_mode import LiveMode


if __name__ == "__main__":
    sys.exit(run_main(LiveMode, default_width=1920, default_height=1200))