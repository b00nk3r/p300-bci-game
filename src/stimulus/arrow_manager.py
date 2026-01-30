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
    """
    
    def __init__(self, config: TriggerConfig):
        self.config = config
        self._file = None
        self._start_time: float = 0.0
        
    def start_session(self):
        """Start a new recording session"""
        self._start_time = time.perf_counter()
        
        if self.config.method == "file":
            # Ensure directory exists
            self.config.trigger_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Open file (append mode)
            self._file = open(self.config.trigger_file, 'a')
            self._file.write(f"# Session started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._file.write("# timestamp_ms, trigger_code, label\n")
            self._file.flush()
            
    def stop_session(self):
        """End the recording session"""
        if self._file:
            self._file.write(f"# Session ended\n\n")
            self._file.close()
            self._file = None
            
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
            self._file.write(f"{timestamp_ms:.3f}, {code}, {label}\n")
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
    
    def __init__(self, config: Config):
        """
        Initialize arrow manager.
        
        Args:
            config: Main application configuration
        """
        self.config = config
        
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
        self.triggers.start_session()
        
    def shutdown(self):
        """Clean up resources"""
        self.triggers.stop_session()
        
    def set_callbacks(
        self,
        on_selection_complete: Optional[Callable[[SelectionResult], None]] = None,
        on_state_change: Optional[Callable[[SelectionState], None]] = None,
    ):
        """
        Set callback functions.
        
        Args:
            on_selection_complete: Called when selection finishes
            on_state_change: Called when state changes
        """
        self._on_selection_complete = on_selection_complete
        self._on_state_change = on_state_change
        
    def start_selection(self):
        """Begin a new BCI selection (start flashing arrows)"""
        if self._state != SelectionState.IDLE:
            return False
            
        # Reset state
        self._flash_states = {d: False for d in Direction.all()}
        self._selected_direction = None
        self._selection_start_time = time.perf_counter()
        
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
        
        # Send trial end trigger
        self.triggers.send_trial_end()
        
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
        
    def _handle_flash_end(self, direction: Direction, sequence: int, time_ms: float):
        """Called when a flash ends"""
        self._flash_states[direction] = False
        
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
        
        # In a real BCI, we'd wait for MATLAB classifier result here
        # For now, we'll need external input or simulation
        
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
            text = font.render(line, True, (50, 50, 50))
            screen.blit(text, (10, y))
            y += 30
            
        pygame.display.flip()
        clock.tick(config.display.fps)
        
    manager.shutdown()
    pygame.quit()
    

if __name__ == "__main__":
    demo()