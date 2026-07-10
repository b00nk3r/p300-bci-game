import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import TriggerConfig, Direction

class TriggerManager:
    
    def __init__(self, config: TriggerConfig):
        self.config = config
        self._file = None
        self._start_time: float = 0.0
        self._session_filepath: Optional[Path] = None
        self._current_target: str = "NONE"
        
    def start_session(
        self,
        flash_duration_ms: Optional[int] = None,
        isi_ms: Optional[int] = None,
        soa_ms: Optional[int] = None,
        num_sequences: Optional[int] = None,
        inter_sequence_pause_ms: Optional[int] = None,
        flash_pattern: Optional[str] = None,
        color_scheme: Optional[str] = None,
        flash_rate_hz: Optional[float] = None,
    ):
        self._start_time = time.perf_counter()
        self._current_target = "NONE"
        session_start_dt = datetime.now()
        
        if self.config.method == "file":
            output_dir = self.config.trigger_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"triggers_{timestamp}.txt"
            self._session_filepath = output_dir / filename
            
            self._file = open(self._session_filepath, 'w')
            self._file.write("=" * 70 + "\n")
            self._file.write("P300 BCI TRIGGER LOG\n")
            self._file.write("=" * 70 + "\n\n")
            
            session_start_str = session_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._file.write(f"Session started: {session_start_str}\n\n")
            
            self._file.write("SESSION PARAMETERS\n")
            self._file.write("-" * 40 + "\n")
            if flash_duration_ms is not None:
                self._file.write(f"Flash Duration:      {flash_duration_ms} ms\n")
            if isi_ms is not None:
                self._file.write(f"ISI:                 {isi_ms} ms\n")
            if num_sequences is not None:
                self._file.write(f"Num Sequences:       {num_sequences}\n")
            
            self._file.write("\n")
            self._file.write("TRIGGER EVENTS\n")
            self._file.write("-" * 40 + "\n")
            self._file.write("Format: timestamp_ms, label, current_target\n\n")
            self._file.flush()
            
    def stop_session(self):
        if self._file:
            self._file.write("\n")
            self._file.write("=" * 70 + "\n")
            session_end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._file.write(f"Session ended: {session_end_str}\n")
            self._file.write("=" * 70 + "\n")
            self._file.close()
            self._file = None
            self._current_target = "NONE"
            if self._session_filepath:
                print(f"Triggers saved: {self._session_filepath}")
            self._session_filepath = None

    def set_current_target(self, target: Optional[Direction]):
        if target is None:
            self._current_target = "NONE"
        else:
            self._current_target = target.value.upper()
            
    def send(self, code: int, label: str = ""):
        if not self.config.enabled:
            return
            
        timestamp_ms = (time.perf_counter() - self._start_time) * 1000
        
        if self.config.method == "file" and self._file:
            self._file.write(f"{timestamp_ms:.3f}, {label}, {self._current_target}\n")
            self._file.flush()
        
    def send_flash(self, direction: Direction):
        code_map = {
            Direction.UP: self.config.FLASH_UP,
            Direction.DOWN: self.config.FLASH_DOWN,
            Direction.LEFT: self.config.FLASH_LEFT,
            Direction.RIGHT: self.config.FLASH_RIGHT,
        }
        code = code_map.get(direction, 0)
        self.send(code, f"flash_{direction.value}")
        
    def send_trial_start(self):
        self.send(self.config.TRIAL_START, "trial_start")

    def send_trial_end(self):
        self.send(self.config.TRIAL_END, "trial_end")
