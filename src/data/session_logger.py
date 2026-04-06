"""
Session Logger
==============
Records BCI session data for later analysis.

Saves:
- Session parameters (flash duration, ISI, sequences, etc.)
- Session start timestamp
- Each flash event with exact timestamp and direction

Output format is a plain text file for easy parsing.
"""

import os
import time
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from config import Direction


@dataclass
class FlashRecord:
    """Record of a single flash event"""
    timestamp_ms: float          # Milliseconds since session start
    absolute_time: float         # Unix timestamp (seconds)
    direction: str               # up, down, left, right
    sequence: int                # Sequence number (0-indexed)
    event_type: str              # "flash_start" or "flash_end"


@dataclass
class SessionData:
    """Complete session data"""
    # Session identification
    session_id: str = ""
    session_start_time: float = 0.0  # Unix timestamp
    session_start_datetime: str = ""  # Human readable
    
    # Parameters
    flash_duration_ms: int = 100
    isi_ms: int = 125
    soa_ms: int = 225
    num_sequences: int = 10
    inter_sequence_pause_ms: int = 200
    flash_pattern: str = "RANDOM"
    color_scheme: str = "GRAY_WHITE"
    
    # Flash events
    flash_events: List[FlashRecord] = field(default_factory=list)
    
    # Session end
    session_end_time: float = 0.0
    session_end_datetime: str = ""
    total_duration_ms: float = 0.0
    
    # Selection result (if applicable)
    target_direction: Optional[str] = None
    selected_direction: Optional[str] = None


class SessionLogger:
    """
    Logs BCI session data to text files.
    
    Usage:
        logger = SessionLogger(output_dir="sessions")
        
        # Start session with parameters
        logger.start_session(
            flash_duration_ms=100,
            isi_ms=125,
            num_sequences=10,
            ...
        )
        
        # Log each flash event
        logger.log_flash_start(Direction.UP, sequence=0, timestamp_ms=0.0)
        logger.log_flash_end(Direction.UP, sequence=0, timestamp_ms=100.0)
        
        # End and save session
        logger.end_session(selected_direction=Direction.UP)
    """
    
    def __init__(self, output_dir: str = "sessions"):
        """
        Initialize session logger.
        
        Args:
            output_dir: Directory to save session files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_session: Optional[SessionData] = None
        self._session_active: bool = False
        
    def start_session(
        self,
        flash_duration_ms: int = 100,
        isi_ms: int = 125,
        num_sequences: int = 10,
        inter_sequence_pause_ms: int = 200,
        flash_pattern: str = "RANDOM",
        color_scheme: str = "GRAY_WHITE",
        target_direction: Optional[Direction] = None,
    ):
        """
        Start a new session.
        
        Args:
            flash_duration_ms: Flash duration in milliseconds
            isi_ms: Inter-stimulus interval in milliseconds
            num_sequences: Number of sequences per selection
            inter_sequence_pause_ms: Pause between sequences
            flash_pattern: RANDOM or SEQUENTIAL
            color_scheme: Color scheme name
            target_direction: Intended target direction for this session/trial
        """
        now = time.time()
        now_dt = datetime.now()
        
        # Create unique session ID
        session_id = now_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        self._current_session = SessionData(
            session_id=session_id,
            session_start_time=now,
            session_start_datetime=now_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            flash_duration_ms=flash_duration_ms,
            isi_ms=isi_ms,
            soa_ms=flash_duration_ms + isi_ms,
            num_sequences=num_sequences,
            inter_sequence_pause_ms=inter_sequence_pause_ms,
            flash_pattern=flash_pattern,
            color_scheme=color_scheme,
            target_direction=target_direction.value if target_direction else None,
        )
        
        self._session_active = True
        print(f"Session started: {session_id}")
        
    def log_flash_start(
        self, 
        direction: Direction, 
        sequence: int, 
        timestamp_ms: float
    ):
        """
        Log a flash start event.
        
        Args:
            direction: Direction of the flash
            sequence: Sequence number (0-indexed)
            timestamp_ms: Milliseconds since session start
        """
        if not self._session_active or self._current_session is None:
            return
            
        record = FlashRecord(
            timestamp_ms=timestamp_ms,
            absolute_time=time.time(),
            direction=direction.value,
            sequence=sequence,
            event_type="flash_start"
        )
        self._current_session.flash_events.append(record)
        
    def log_flash_end(
        self, 
        direction: Direction, 
        sequence: int, 
        timestamp_ms: float
    ):
        """
        Log a flash end event.
        
        Args:
            direction: Direction of the flash
            sequence: Sequence number (0-indexed)
            timestamp_ms: Milliseconds since session start
        """
        if not self._session_active or self._current_session is None:
            return
            
        record = FlashRecord(
            timestamp_ms=timestamp_ms,
            absolute_time=time.time(),
            direction=direction.value,
            sequence=sequence,
            event_type="flash_end"
        )
        self._current_session.flash_events.append(record)
        
    def end_session(self, selected_direction: Optional[Direction] = None) -> Optional[str]:
        """
        End the current session and save to file.
        
        Args:
            selected_direction: The direction that was selected (if any)
            
        Returns:
            Path to the saved file, or None if no session was active
        """
        if not self._session_active or self._current_session is None:
            return None
            
        now = time.time()
        now_dt = datetime.now()
        
        self._current_session.session_end_time = now
        self._current_session.session_end_datetime = now_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._current_session.total_duration_ms = (
            now - self._current_session.session_start_time
        ) * 1000.0
        
        if selected_direction:
            self._current_session.selected_direction = selected_direction.value
            
        # Save to file
        filepath = self._save_session()
        
        self._session_active = False
        self._current_session = None
        
        return filepath
        
    def cancel_session(self):
        """Cancel the current session without saving"""
        self._session_active = False
        self._current_session = None
        print("Session cancelled")
        
    def _save_session(self) -> str:
        """Save session data to text file"""
        if self._current_session is None:
            return ""
            
        session = self._current_session
        filename = f"session_{session.session_id}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            # Header
            f.write("=" * 70 + "\n")
            f.write("P300 BCI SESSION LOG\n")
            f.write("=" * 70 + "\n\n")
            
            # Session Info
            f.write("SESSION INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Session ID:          {session.session_id}\n")
            f.write(f"Start Time:          {session.session_start_datetime}\n")
            f.write(f"End Time:            {session.session_end_datetime}\n")
            f.write(f"Total Duration:      {session.total_duration_ms:.2f} ms\n")
            if session.target_direction:
                f.write(f"Target Direction:    {session.target_direction}\n")
            if session.selected_direction:
                f.write(f"Selected Direction:  {session.selected_direction}\n")
            f.write("\n")
            
            # Parameters
            f.write("SESSION PARAMETERS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Flash Duration:      {session.flash_duration_ms} ms\n")
            f.write(f"ISI:                 {session.isi_ms} ms\n")
            f.write(f"SOA:                 {session.soa_ms} ms\n")
            f.write(f"Num Sequences:       {session.num_sequences}\n")
            f.write(f"Inter-Seq Pause:     {session.inter_sequence_pause_ms} ms\n")
            f.write(f"Flash Pattern:       {session.flash_pattern}\n")
            f.write(f"Color Scheme:        {session.color_scheme}\n")
            f.write("\n")
            
            # Flash Events Header
            f.write("FLASH EVENTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Events:        {len(session.flash_events)}\n")
            f.write("\n")
            
            # Column headers
            f.write(f"{'Timestamp_ms':>12}  {'Absolute_Time':>18}  {'Direction':>8}  ")
            f.write(f"{'Sequence':>8}  {'Event_Type':>12}\n")
            f.write("-" * 70 + "\n")
            
            # Flash events
            for event in session.flash_events:
                f.write(f"{event.timestamp_ms:>12.3f}  ")
                f.write(f"{event.absolute_time:>18.6f}  ")
                f.write(f"{event.direction:>8}  ")
                f.write(f"{event.sequence:>8}  ")
                f.write(f"{event.event_type:>12}\n")
                
            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write("END OF SESSION LOG\n")
            f.write("=" * 70 + "\n")
            
        print(f"Session saved: {filepath}")
        return str(filepath)
        
    @property
    def is_active(self) -> bool:
        """Whether a session is currently active"""
        return self._session_active
        
    @property
    def current_session_id(self) -> Optional[str]:
        """Get current session ID"""
        if self._current_session:
            return self._current_session.session_id
        return None


# =============================================================================
# Testing
# =============================================================================

def demo():
    """Demo the session logger"""
    import random
    
    logger = SessionLogger(output_dir="test_sessions")
    
    # Start session
    logger.start_session(
        flash_duration_ms=100,
        isi_ms=125,
        num_sequences=3,
        flash_pattern="RANDOM",
        color_scheme="GRAY_WHITE"
    )
    
    # Simulate flash events
    current_time = 0.0
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    
    for seq in range(3):
        random.shuffle(directions)
        for direction in directions:
            # Flash start
            logger.log_flash_start(direction, seq, current_time)
            time.sleep(0.01)  # Small delay to get different absolute times
            
            # Flash end
            current_time += 100.0
            logger.log_flash_end(direction, seq, current_time)
            time.sleep(0.01)
            
            # Next SOA
            current_time += 125.0
            
        # Inter-sequence pause
        current_time += 200.0
        
    # End session
    filepath = logger.end_session(selected_direction=Direction.UP)
    
    print(f"\nSession file saved to: {filepath}")
    
    # Show file contents
    print("\nFile contents:")
    print("-" * 40)
    with open(filepath) as f:
        print(f.read())


if __name__ == "__main__":
    demo()
