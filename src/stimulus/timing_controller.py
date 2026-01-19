"""
Timing Controller
=================
Precise timing for stimulus presentation using high-resolution timer.
"""

import time
import random
from typing import Callable, Optional, List
from dataclasses import dataclass
from enum import Enum, auto

from config import TimingConfig, Direction


class TimerState(Enum):
    """State of the timing controller"""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETE = auto()


@dataclass
class FlashEvent:
    """Represents a scheduled flash event"""
    time_ms: float          # When to start (relative to sequence start)
    direction: Direction    # Which arrow
    is_flash_on: bool       # True = start flash, False = end flash


class TimingController:
    """
    Controls precise timing of stimulus presentation.
    
    Uses time.perf_counter() for sub-millisecond precision.
    Pre-schedules all events for a selection to minimize runtime overhead.
    """
    
    def __init__(self, config: TimingConfig):
        self.config = config
        
        self._state = TimerState.IDLE
        self._start_time: float = 0.0
        self._events: List[FlashEvent] = []
        self._event_index: int = 0
        
        # Callbacks
        self._on_flash_start: Optional[Callable[[Direction], None]] = None
        self._on_flash_end: Optional[Callable[[Direction], None]] = None
        self._on_sequence_complete: Optional[Callable[[int], None]] = None
        self._on_selection_complete: Optional[Callable[[], None]] = None
        
    def set_callbacks(
        self,
        on_flash_start: Optional[Callable[[Direction], None]] = None,
        on_flash_end: Optional[Callable[[Direction], None]] = None,
        on_sequence_complete: Optional[Callable[[int], None]] = None,
        on_selection_complete: Optional[Callable[[], None]] = None,
    ):
        """Set callback functions for timing events"""
        self._on_flash_start = on_flash_start
        self._on_flash_end = on_flash_end
        self._on_sequence_complete = on_sequence_complete
        self._on_selection_complete = on_selection_complete
        
    def start_selection(self):
        """Start a new selection (begin flashing sequence)"""
        # TODO: Build event schedule
        # TODO: Start timer
        self._state = TimerState.RUNNING
        self._start_time = time.perf_counter()
        self._event_index = 0
        self._build_schedule()
        
    def stop_selection(self):
        """Stop the current selection"""
        self._state = TimerState.IDLE
        
    def update(self) -> Optional[Direction]:
        """
        Update timing and trigger any due events.
        Call this every frame.
        
        Returns:
            Direction that is currently flashing, or None
        """
        if self._state != TimerState.RUNNING:
            return None
            
        # TODO: Check for due events and trigger callbacks
        # TODO: Return currently flashing direction
        return None
        
    def _build_schedule(self):
        """Build the complete event schedule for all sequences"""
        self._events = []
        current_time = 0.0
        
        for seq in range(self.config.num_sequences):
            # Randomize order for this sequence
            directions = list(Direction.all())
            random.shuffle(directions)
            
            for direction in directions:
                # Flash start
                self._events.append(FlashEvent(
                    time_ms=current_time,
                    direction=direction,
                    is_flash_on=True
                ))
                
                # Flash end
                self._events.append(FlashEvent(
                    time_ms=current_time + self.config.flash_duration_ms,
                    direction=direction,
                    is_flash_on=False
                ))
                
                current_time += self.config.soa_ms
                
            # Inter-sequence pause
            current_time += self.config.inter_sequence_pause_ms
            
    def _elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds"""
        return (time.perf_counter() - self._start_time) * 1000.0
