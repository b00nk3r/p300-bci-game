"""
Timing Controller
=================
Precise timing for stimulus presentation using high-resolution timer.

Key features:
- Sub-millisecond precision using time.perf_counter()
- Pre-scheduled events to minimize runtime overhead
- Callback-based architecture for flexibility
- Timing statistics for verification

For P300 BCI, timing jitter should be <10ms for reliable ERP averaging.
"""

import time
import random
from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto

from config import TimingConfig, Direction, FlashPattern


class TimerState(Enum):
    """State of the timing controller"""
    IDLE = auto()       # Not running
    RUNNING = auto()    # Actively processing events
    PAUSED = auto()     # Temporarily paused
    COMPLETE = auto()   # Finished all sequences


@dataclass
class FlashEvent:
    """
    Represents a scheduled flash event.
    
    Attributes:
        time_ms: When this event should occur (relative to start)
        direction: Which arrow this event affects
        is_on: True = flash starts, False = flash ends
        sequence: Which sequence number (0-indexed)
        index: Index within the sequence (0-3 for 4 arrows)
    """
    time_ms: float
    direction: Direction
    is_on: bool
    sequence: int
    index: int


@dataclass
class TimingStats:
    """Statistics about timing performance"""
    total_events: int = 0
    processed_events: int = 0
    timing_errors_ms: List[float] = field(default_factory=list)
    
    @property
    def mean_error_ms(self) -> float:
        if not self.timing_errors_ms:
            return 0.0
        return sum(self.timing_errors_ms) / len(self.timing_errors_ms)
    
    @property
    def max_error_ms(self) -> float:
        if not self.timing_errors_ms:
            return 0.0
        return max(abs(e) for e in self.timing_errors_ms)
    
    @property 
    def is_acceptable(self) -> bool:
        """Check if timing is acceptable for P300 (<10ms jitter)"""
        return self.max_error_ms < 10.0


class TimingController:
    """
    Controls precise timing of stimulus presentation.
    
    Uses time.perf_counter() for sub-millisecond precision.
    Pre-schedules all events to minimize runtime overhead.
    
    Usage:
        controller = TimingController(timing_config)
        controller.set_callbacks(
            on_flash_start=lambda d: print(f"{d} ON"),
            on_flash_end=lambda d: print(f"{d} OFF"),
        )
        
        controller.start()
        
        # In game loop:
        while controller.is_running:
            current_flash = controller.update()
            # Draw arrows based on current_flash
    """
    
    def __init__(self, config: TimingConfig):
        """
        Initialize timing controller.
        
        Args:
            config: Timing configuration with flash duration, ISI, etc.
        """
        self.config = config
        
        # State
        self._state = TimerState.IDLE
        self._start_time: float = 0.0
        self._pause_time: float = 0.0
        self._pause_duration: float = 0.0
        
        # Event schedule
        self._events: List[FlashEvent] = []
        self._event_index: int = 0
        
        # Current flash state
        self._current_flash: Optional[Direction] = None
        self._current_sequence: int = 0
        
        # Callbacks
        self._on_flash_start: Optional[Callable[[Direction, int, float], None]] = None
        self._on_flash_end: Optional[Callable[[Direction, int, float], None]] = None
        self._on_sequence_complete: Optional[Callable[[int], None]] = None
        self._on_selection_complete: Optional[Callable[[], None]] = None
        
        # Statistics
        self._stats = TimingStats()
        
    def set_callbacks(
        self,
        on_flash_start: Optional[Callable[[Direction, int, float], None]] = None,
        on_flash_end: Optional[Callable[[Direction, int, float], None]] = None,
        on_sequence_complete: Optional[Callable[[int], None]] = None,
        on_selection_complete: Optional[Callable[[], None]] = None,
    ):
        """
        Set callback functions for timing events.
        
        Args:
            on_flash_start: Called when flash begins (direction, sequence, timestamp_ms)
            on_flash_end: Called when flash ends (direction, sequence, timestamp_ms)
            on_sequence_complete: Called when a sequence finishes (sequence_number)
            on_selection_complete: Called when all sequences are done
        """
        self._on_flash_start = on_flash_start
        self._on_flash_end = on_flash_end
        self._on_sequence_complete = on_sequence_complete
        self._on_selection_complete = on_selection_complete
        
    def start(self):
        """Start a new selection (begin flashing sequence)"""
        if self._state == TimerState.RUNNING:
            return
            
        # Build event schedule
        self._build_schedule()
        
        # Reset state
        self._event_index = 0
        self._current_flash = None
        self._current_sequence = 0
        self._pause_duration = 0.0
        
        # Reset stats
        self._stats = TimingStats(total_events=len(self._events))
        
        # Start timer
        self._start_time = time.perf_counter()
        self._state = TimerState.RUNNING
        
    def stop(self):
        """Stop the current selection"""
        self._state = TimerState.IDLE
        self._current_flash = None
        
    def pause(self):
        """Pause timing (for settings menu, etc.)"""
        if self._state == TimerState.RUNNING:
            self._pause_time = time.perf_counter()
            self._state = TimerState.PAUSED
            
    def resume(self):
        """Resume from pause"""
        if self._state == TimerState.PAUSED:
            self._pause_duration += time.perf_counter() - self._pause_time
            self._state = TimerState.RUNNING
            
    def update(self) -> Optional[Direction]:
        """
        Update timing and process any due events.
        Call this every frame.
        
        Returns:
            Direction currently flashing, or None if no flash active
        """
        if self._state != TimerState.RUNNING:
            return self._current_flash
            
        # Get current elapsed time
        elapsed_ms = self._elapsed_ms()
        
        # Process all due events
        while self._event_index < len(self._events):
            event = self._events[self._event_index]
            
            if event.time_ms <= elapsed_ms:
                self._process_event(event, elapsed_ms)
                self._event_index += 1
            else:
                # No more events due yet
                break
                
        # Check if complete
        if self._event_index >= len(self._events):
            self._state = TimerState.COMPLETE
            self._current_flash = None
            if self._on_selection_complete:
                self._on_selection_complete()
                
        return self._current_flash
        
    def _process_event(self, event: FlashEvent, actual_time_ms: float):
        """Process a single timing event"""
        # Record timing error
        error = actual_time_ms - event.time_ms
        self._stats.timing_errors_ms.append(error)
        self._stats.processed_events += 1
        
        if event.is_on:
            # Flash start
            self._current_flash = event.direction
            
            if self._on_flash_start:
                self._on_flash_start(event.direction, event.sequence, actual_time_ms)
                
        else:
            # Flash end
            if self._current_flash == event.direction:
                self._current_flash = None
                
            if self._on_flash_end:
                self._on_flash_end(event.direction, event.sequence, actual_time_ms)
                
            # Check for sequence complete (after last flash ends in sequence)
            if event.index == 3:  # Last arrow in sequence (0-indexed)
                self._current_sequence = event.sequence + 1
                if self._on_sequence_complete:
                    self._on_sequence_complete(event.sequence)
                    
    def _build_schedule(self):
        """Build the complete event schedule for all sequences"""
        self._events = []
        current_time = 0.0
        
        flash_duration = self.config.flash_duration_ms
        soa = self.config.soa_ms
        
        for seq in range(self.config.num_sequences):
            # Get flash order for this sequence
            directions = self._get_flash_order(seq)
            
            for idx, direction in enumerate(directions):
                # Flash ON event
                self._events.append(FlashEvent(
                    time_ms=current_time,
                    direction=direction,
                    is_on=True,
                    sequence=seq,
                    index=idx
                ))
                
                # Flash OFF event
                self._events.append(FlashEvent(
                    time_ms=current_time + flash_duration,
                    direction=direction,
                    is_on=False,
                    sequence=seq,
                    index=idx
                ))
                
                # Move to next flash time (SOA)
                current_time += soa
                
            # Add inter-sequence pause
            current_time += self.config.inter_sequence_pause_ms
            
    def _get_flash_order(self, sequence: int) -> List[Direction]:
        """
        Get the order of flashes for a sequence.
        
        Args:
            sequence: Sequence number (for potential pattern variations)
            
        Returns:
            List of directions in flash order
        """
        directions = list(Direction.all())
        
        if self.config.flash_pattern == FlashPattern.RANDOM:
            random.shuffle(directions)
        # SEQUENTIAL keeps default order: UP, DOWN, LEFT, RIGHT
        
        return directions
        
    def _elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds (excluding pauses)"""
        if self._state == TimerState.PAUSED:
            return (self._pause_time - self._start_time - self._pause_duration) * 1000.0
        return (time.perf_counter() - self._start_time - self._pause_duration) * 1000.0
        
    @property
    def is_running(self) -> bool:
        """Whether the controller is actively running"""
        return self._state == TimerState.RUNNING
        
    @property
    def is_complete(self) -> bool:
        """Whether all sequences have finished"""
        return self._state == TimerState.COMPLETE
        
    @property
    def current_sequence(self) -> int:
        """Current sequence number (0-indexed)"""
        return self._current_sequence
        
    @property
    def progress(self) -> float:
        """Progress through all sequences (0.0 to 1.0)"""
        if not self._events:
            return 0.0
        return self._event_index / len(self._events)
        
    @property
    def stats(self) -> TimingStats:
        """Get timing statistics"""
        return self._stats
        
    def get_expected_duration_ms(self) -> float:
        """Calculate expected total duration in milliseconds"""
        num_flashes_per_seq = 4  # 4 directions
        seq_duration = num_flashes_per_seq * self.config.soa_ms
        total = self.config.num_sequences * (
            seq_duration + self.config.inter_sequence_pause_ms
        )
        return total


# =============================================================================
# Testing / Demo
# =============================================================================

def demo():
    """Run a demo of the timing controller (console output)"""
    from config import Config
    
    config = Config()
    config.timing.num_sequences = 2  # Short demo
    
    controller = TimingController(config.timing)
    
    # Track events for verification
    events_log = []
    
    def on_flash_start(direction, seq, time_ms):
        events_log.append(('ON', direction.value, seq, time_ms))
        print(f"  {time_ms:8.1f}ms: {direction.value:5} ON  (seq {seq})")
        
    def on_flash_end(direction, seq, time_ms):
        events_log.append(('OFF', direction.value, seq, time_ms))
        print(f"  {time_ms:8.1f}ms: {direction.value:5} OFF")
        
    def on_sequence_complete(seq):
        print(f"  --- Sequence {seq} complete ---")
        
    def on_selection_complete():
        print("  === Selection complete ===")
        
    controller.set_callbacks(
        on_flash_start=on_flash_start,
        on_flash_end=on_flash_end,
        on_sequence_complete=on_sequence_complete,
        on_selection_complete=on_selection_complete,
    )
    
    print("TimingController Demo")
    print(f"  Flash duration: {config.timing.flash_duration_ms}ms")
    print(f"  ISI: {config.timing.isi_ms}ms")
    print(f"  SOA: {config.timing.soa_ms}ms")
    print(f"  Sequences: {config.timing.num_sequences}")
    print(f"  Expected duration: {controller.get_expected_duration_ms():.0f}ms")
    print()
    print("Starting selection...")
    print()
    
    controller.start()
    
    # Run until complete
    while controller.is_running:
        controller.update()
        time.sleep(0.001)  # 1ms sleep to avoid busy-waiting too hard
        
    print()
    print("Timing Statistics:")
    stats = controller.stats
    print(f"  Total events: {stats.total_events}")
    print(f"  Processed: {stats.processed_events}")
    print(f"  Mean error: {stats.mean_error_ms:.3f}ms")
    print(f"  Max error: {stats.max_error_ms:.3f}ms")
    print(f"  Acceptable: {'✓ YES' if stats.is_acceptable else '✗ NO'}")
    

if __name__ == "__main__":
    demo()