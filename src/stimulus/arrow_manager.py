"""
Arrow Manager
=============
Coordinates arrow rendering and timing for P300 BCI interface.

This is the main interface for the stimulus presentation system.
It brings together:
- ArrowRenderer for visual display
- TimingController for precise timing
- Trigger output for EEG synchronization
- State management for the selection process
"""

import pygame
import time
from datetime import datetime
from typing import Optional, Callable, Dict, List
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path

from config import Config, Direction, TriggerConfig


class SelectionState(Enum):
    """State of the BCI selection process"""
    IDLE = auto()        # Waiting to start
    FLASHING = auto()    # Arrows are flashing
    PROCESSING = auto()  # Waiting for classifier result
    FEEDBACK = auto()    # Showing selection feedback
    COMPLETE = auto()    # Selection finished


@dataclass
class SelectionResult:
    """Result of a completed selection"""
    direction: Optional[Direction]  # Selected direction (None if timeout/cancelled)
    confidence: float               # Classifier confidence (0-1)
    num_sequences: int              # How many sequences were completed
    duration_ms: float              # Total selection duration
    timing_stats: dict              # Timing performance stats


class TriggerManager:
    """
    Manages trigger output for EEG synchronization.
    
    Supports multiple output methods:
    - file: Write to text file (for MATLAB to read)
    - lsl: Lab Streaming Layer (real-time)
    - serial: Serial port
    
    Each session creates a separate file with timestamp in the filename.
    """
    
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
        """
        Start a new recording session with unique filename.
        
        Args:
            flash_duration_ms: Flash duration in milliseconds
            isi_ms: Inter-stimulus interval in milliseconds
            soa_ms: Stimulus onset asynchrony in milliseconds
            num_sequences: Number of sequences per selection
            inter_sequence_pause_ms: Pause between sequences in milliseconds
            flash_pattern: Flash pattern (RANDOM or SEQUENTIAL)
            color_scheme: Color scheme name
            flash_rate_hz: Flash rate in Hz
        """
        self._start_time = time.perf_counter()
        self._current_target = "NONE"
        session_start_dt = datetime.now()
        
        if self.config.method == "file":
            # Ensure directory exists
            output_dir = self.config.trigger_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create unique filename with timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"triggers_{timestamp}.txt"
            self._session_filepath = output_dir / filename
            
            # Open new file for this session
            self._file = open(self._session_filepath, 'w')
            self._file.write("=" * 70 + "\n")
            self._file.write("P300 BCI TRIGGER LOG\n")
            self._file.write("=" * 70 + "\n\n")
            
            # Session info
            session_start_str = session_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._file.write(f"Session started: {session_start_str}\n\n")
            
            # Session parameters
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
        """End the recording session"""
        if self._file:
            self._file.write(f"\n")
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
        """
        Set the currently attended target for trigger logging.

        Args:
            target: Target direction, or None when no target is active
        """
        if target is None:
            self._current_target = "NONE"
        else:
            self._current_target = target.value.upper()
            
    def send(self, code: int, label: str = ""):
        """
        Send a trigger.
        
        Args:
            code: Trigger code (integer)
            label: Human-readable label
        """
        if not self.config.enabled:
            return
            
        timestamp_ms = (time.perf_counter() - self._start_time) * 1000
        
        if self.config.method == "file" and self._file:
            self._file.write(f"{timestamp_ms:.3f}, {label}, {self._current_target}\n")
            self._file.flush()
            
        # TODO: Add LSL support
        # TODO: Add serial port support
        
    def send_flash(self, direction: Direction):
        """Send trigger for arrow flash"""
        code_map = {
            Direction.UP: self.config.FLASH_UP,
            Direction.DOWN: self.config.FLASH_DOWN,
            Direction.LEFT: self.config.FLASH_LEFT,
            Direction.RIGHT: self.config.FLASH_RIGHT,
        }
        code = code_map.get(direction, 0)
        self.send(code, f"flash_{direction.value}")
        
    def send_trial_start(self):
        """Send trial start trigger"""
        self.send(self.config.TRIAL_START, "trial_start")
        
    def send_trial_end(self):
        """Send trial end trigger"""
        self.send(self.config.TRIAL_END, "trial_end")
        
    def send_selection(self, direction: Direction):
        """Send selection trigger"""
        self.send(self.config.SELECTION, f"selection_{direction.value}")


class ArrowManager:
    """
    Main interface for the P300 arrow stimulus system.
    
    Coordinates:
    - Arrow rendering (visual display)
    - Flash timing (precise stimulus presentation)
    - Trigger output (EEG synchronization)
    - Selection state (BCI workflow)
    
    Usage:
        manager = ArrowManager(config)
        manager.initialize(screen_width, screen_height)
        manager.set_selection_callback(on_selection)
        
        # Start BCI selection
        manager.start_selection()
        
        # In game loop:
        manager.update()
        manager.draw(screen)
    """
    
    def __init__(self, config: Config, bci_controller=None):
        """
        Initialize arrow manager.
        
        Args:
            config: Main application configuration
            bci_controller: Optional BCIController for real-time EEG classification.
                            When provided, flash events are forwarded to the BCI
                            pipeline and classification runs automatically at
                            the end of each trial.
        """
        self.config = config
        self.bci_controller = bci_controller
        
        # Import here to avoid circular imports
        from src.stimulus.arrow_renderer import ArrowRenderer
        from src.stimulus.timing_controller import TimingController
        
        # Create sub-components
        self.renderer = ArrowRenderer(config.arrows, config.layout)
        self.timing = TimingController(config.timing)
        self.triggers = TriggerManager(config.triggers)
        
        # State
        self._state = SelectionState.IDLE
        self._flash_states: Dict[Direction, bool] = {d: False for d in Direction.all()}
        
        # Selection tracking
        self._selection_start_time: float = 0.0
        self._selected_direction: Optional[Direction] = None
        self._feedback_start_time: float = 0.0
        
        # Callbacks
        self._on_selection_complete: Optional[Callable[[SelectionResult], None]] = None
        self._on_state_change: Optional[Callable[[SelectionState], None]] = None
        self._on_flash_start: Optional[Callable[[Direction, int, float], None]] = None
        self._on_flash_end: Optional[Callable[[Direction, int, float], None]] = None
        
        # Setup timing callbacks
        self.timing.set_callbacks(
            on_flash_start=self._handle_flash_start,
            on_flash_end=self._handle_flash_end,
            on_sequence_complete=self._handle_sequence_complete,
            on_selection_complete=self._handle_timing_complete,
        )
        
    def initialize(self, screen_width: int, screen_height: int):
        """
        Initialize with screen dimensions.
        Must be called after pygame display is created.
        
        Args:
            screen_width: Display width in pixels
            screen_height: Display height in pixels
        """
        self.renderer.initialize(screen_width, screen_height)
        
    def shutdown(self):
        """Clean up resources"""
        # Close any active trigger session
        if self.triggers._file:
            self.triggers.stop_session()
        
    def set_callbacks(
        self,
        on_selection_complete: Optional[Callable[[SelectionResult], None]] = None,
        on_state_change: Optional[Callable[[SelectionState], None]] = None,
        on_flash_start: Optional[Callable[[Direction, int, float], None]] = None,
        on_flash_end: Optional[Callable[[Direction, int, float], None]] = None,
    ):
        """
        Set callback functions.
        
        Args:
            on_selection_complete: Called when selection finishes
            on_state_change: Called when state changes
            on_flash_start: Called when flash begins (direction, sequence, timestamp_ms)
            on_flash_end: Called when flash ends (direction, sequence, timestamp_ms)
        """
        self._on_selection_complete = on_selection_complete
        self._on_state_change = on_state_change
        self._on_flash_start = on_flash_start
        self._on_flash_end = on_flash_end
        
    def start_selection(self):
        """Begin a new BCI selection (start flashing arrows)"""
        if self._state != SelectionState.IDLE:
            return False
            
        # Reset state
        self._flash_states = {d: False for d in Direction.all()}
        self._selected_direction = None
        self._selection_start_time = time.perf_counter()
        
        if self.bci_controller is not None:
            self.bci_controller.begin_trial()
        
        # Start new trigger session (creates new file with parameters)
        self.triggers.start_session(
            flash_duration_ms=self.config.timing.flash_duration_ms,
            isi_ms=self.config.timing.isi_ms,
            soa_ms=self.config.timing.soa_ms,
            num_sequences=self.config.timing.num_sequences,
            inter_sequence_pause_ms=self.config.timing.inter_sequence_pause_ms,
            flash_pattern=self.config.timing.flash_pattern.name,
            color_scheme=self.config.arrows.color_scheme.name,
            flash_rate_hz=self.config.timing.flash_rate_hz,
        )
        
        # Send trial start trigger
        self.triggers.send_trial_start()
        
        # Start timing controller
        self.timing.start()
        
        # Update state
        self._set_state(SelectionState.FLASHING)
        
        return True
        
    def stop_selection(self):
        """Stop the current selection"""
        if self._state == SelectionState.IDLE:
            return
            
        self.timing.stop()
        self._flash_states = {d: False for d in Direction.all()}
        
        # Send trial end trigger and close session
        self.triggers.send_trial_end()
        self.triggers.stop_session()
        
        self._set_state(SelectionState.IDLE)
        
    def update(self):
        """
        Update state. Call every frame.
        """
        if self._state == SelectionState.FLASHING:
            # Update timing controller
            current_flash = self.timing.update()
            
            # Update flash states based on timing
            # (already handled via callbacks, but sync here too)
            
        elif self._state == SelectionState.FEEDBACK:
            # Check if feedback duration has elapsed
            elapsed = (time.perf_counter() - self._feedback_start_time) * 1000
            if elapsed >= self.config.timing.feedback_duration_ms:
                self._complete_selection()
                
    def draw(self, screen: pygame.Surface):
        """
        Draw arrows. Call every frame.
        
        Args:
            screen: Pygame surface to draw on
        """
        self.renderer.draw(screen, self._flash_states)
        
    def simulate_selection(self, direction: Direction):
        """
        Simulate a classifier selection (for testing without EEG).
        
        Args:
            direction: The "selected" direction
        """
        if self._state != SelectionState.PROCESSING:
            # If still flashing, stop and go to processing
            if self._state == SelectionState.FLASHING:
                self.timing.stop()
                self._set_state(SelectionState.PROCESSING)
                
        self._handle_classifier_result(direction, confidence=1.0)
        
    def _set_state(self, new_state: SelectionState):
        """Update state and notify callback"""
        old_state = self._state
        self._state = new_state
        
        if self._on_state_change and old_state != new_state:
            self._on_state_change(new_state)
            
    def _handle_flash_start(self, direction: Direction, sequence: int, time_ms: float):
        """Called when a flash begins"""
        self._flash_states[direction] = True
        
        # Send trigger
        self.triggers.send_flash(direction)
        
        # Record flash for BCI pipeline (uses LSL clock, not perf_counter)
        if self.bci_controller is not None:
            self.bci_controller.record_flash(direction.value)
        
        # Call external callback for logging
        if self._on_flash_start:
            self._on_flash_start(direction, sequence, time_ms)
        
    def _handle_flash_end(self, direction: Direction, sequence: int, time_ms: float):
        """Called when a flash ends"""
        self._flash_states[direction] = False
        
        # Call external callback for logging
        if self._on_flash_end:
            self._on_flash_end(direction, sequence, time_ms)
        
    def _handle_sequence_complete(self, sequence: int):
        """Called when a sequence finishes"""
        # Could update progress display here
        pass
        
    def _handle_timing_complete(self):
        """Called when all sequences are done"""
        # Send trial end trigger
        self.triggers.send_trial_end()
        
        # Move to processing state (waiting for classifier)
        self._set_state(SelectionState.PROCESSING)
        
        # If BCI controller is available, classify immediately
        if self.bci_controller is not None and self.bci_controller.is_connected():
            try:
                result = self.bci_controller.end_trial()
                if result and result.get("predicted_direction"):
                    direction_str = result["predicted_direction"]
                    dir_map = {
                        "up": Direction.UP,
                        "down": Direction.DOWN,
                        "left": Direction.LEFT,
                        "right": Direction.RIGHT,
                    }
                    direction = dir_map.get(direction_str)
                    if direction is not None:
                        scores = result.get("direction_scores", {})
                        n_clean = result.get("n_clean_epochs", 0)
                        n_total = result.get("n_total_epochs", 0)
                        print(f"BCI classification: {direction_str} "
                              f"({n_clean}/{n_total} clean epochs)")
                        for d, s in sorted(scores.items()):
                            print(f"  {d}: {s:.4f}")
                        self._handle_classifier_result(direction, confidence=1.0)
                        return
                else:
                    print("BCI classification returned no result, "
                          "waiting for keyboard input")
            except Exception as e:
                print(f"BCI classification error: {e}, "
                      "waiting for keyboard input")
        
    def _handle_classifier_result(self, direction: Direction, confidence: float):
        """
        Handle classifier result from MATLAB.
        
        Args:
            direction: Classified direction
            confidence: Classifier confidence (0-1)
        """
        self._selected_direction = direction
        
        # Send selection trigger
        self.triggers.send_selection(direction)
        
        # Show feedback
        self._flash_states = {d: False for d in Direction.all()}
        self._flash_states[direction] = True  # Highlight selected
        self._feedback_start_time = time.perf_counter()
        
        self._set_state(SelectionState.FEEDBACK)
        
    def _complete_selection(self):
        """Finish the selection process"""
        # Calculate duration
        duration_ms = (time.perf_counter() - self._selection_start_time) * 1000
        
        # Create result
        result = SelectionResult(
            direction=self._selected_direction,
            confidence=1.0,  # TODO: Get from classifier
            num_sequences=self.timing.current_sequence,
            duration_ms=duration_ms,
            timing_stats={
                'mean_error_ms': self.timing.stats.mean_error_ms,
                'max_error_ms': self.timing.stats.max_error_ms,
                'acceptable': self.timing.stats.is_acceptable,
            }
        )
        
        # Close trigger session
        self.triggers.stop_session()
        
        # Reset state
        self._flash_states = {d: False for d in Direction.all()}
        self._set_state(SelectionState.IDLE)
        
        # Notify callback
        if self._on_selection_complete:
            self._on_selection_complete(result)
            
    @property
    def state(self) -> SelectionState:
        """Current selection state"""
        return self._state
        
    @property
    def is_active(self) -> bool:
        """Whether a selection is in progress"""
        return self._state in (SelectionState.FLASHING, SelectionState.PROCESSING, 
                               SelectionState.FEEDBACK)
        
    @property
    def progress(self) -> float:
        """Progress through current selection (0.0 to 1.0)"""
        return self.timing.progress
        
    def get_panel_rect(self) -> Optional[pygame.Rect]:
        """Get the arrow panel rectangle (for game layout)"""
        return self.renderer.get_panel_rect()


# =============================================================================
# Testing / Demo
# =============================================================================

def demo():
    """Run a visual demo of the complete arrow system"""
    pygame.init()
    
    config = Config()
    config.display.width = 1024
    config.display.height = 768
    config.timing.num_sequences = 3  # Short demo
    
    screen = pygame.display.set_mode(
        (config.display.width, config.display.height)
    )
    pygame.display.set_caption("Arrow Manager Demo - Press SPACE to start")
    
    clock = pygame.time.Clock()
    
    # Create manager
    manager = ArrowManager(config)
    manager.initialize(config.display.width, config.display.height)
    
    # Track state changes
    def on_state_change(state):
        print(f"State: {state.name}")
        
    def on_selection_complete(result):
        print(f"Selection complete!")
        print(f"  Direction: {result.direction.value if result.direction else 'None'}")
        print(f"  Duration: {result.duration_ms:.0f}ms")
        print(f"  Timing OK: {result.timing_stats['acceptable']}")
        
    manager.set_callbacks(
        on_state_change=on_state_change,
        on_selection_complete=on_selection_complete,
    )
    
    running = True
    font = pygame.font.Font(None, 32)
    
    print("Arrow Manager Demo")
    print("  SPACE: Start/stop selection")
    print("  1-4: Simulate selection (Up/Down/Left/Right)")
    print("  ESC: Quit")
    print()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
                elif event.key == pygame.K_SPACE:
                    if manager.is_active:
                        manager.stop_selection()
                    else:
                        manager.start_selection()
                        
                # Simulate selections
                elif event.key == pygame.K_1:
                    manager.simulate_selection(Direction.UP)
                elif event.key == pygame.K_2:
                    manager.simulate_selection(Direction.DOWN)
                elif event.key == pygame.K_3:
                    manager.simulate_selection(Direction.LEFT)
                elif event.key == pygame.K_4:
                    manager.simulate_selection(Direction.RIGHT)
                    
        # Update
        manager.update()
        
        # Draw
        screen.fill(config.display.background_color)
        manager.draw(screen)
        
        # Draw status
        status_lines = [
            f"State: {manager.state.name}",
            f"Progress: {manager.progress * 100:.0f}%",
            "",
            "SPACE: Start/Stop",
            "1-4: Simulate selection",
        ]
        
        y = 10
        for line in status_lines:
            text = font.render(line, True, (100, 100, 100))
            screen.blit(text, (10, y))
            y += 30
            
        pygame.display.flip()
        clock.tick(config.display.fps)
        
    manager.shutdown()
    pygame.quit()
    

if __name__ == "__main__":
    demo()