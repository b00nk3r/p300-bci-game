"""
Auto Play Controller
====================
Controls the game automatically using generated directions for testing purposes.

Features:
- Generates random valid directions based on current game state
- Logs all generated directions to a data file
- Triggers moves automatically after the previous move completes
- Uses intelligent direction selection (prefers valid moves, avoids backtracking)
"""

import random
import time
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import Direction


@dataclass
class AutoPlayConfig:
    """Configuration for auto-play controller"""
    # Timing
    delay_between_moves_ms: float = 500.0  # Delay after move completes before next
    
    # Direction generation
    prefer_valid_moves: bool = True       # Prefer directions that lead to valid cells
    avoid_backtracking: bool = True       # Avoid immediately reversing direction
    random_seed: Optional[int] = None     # Set for reproducible sequences
    
    # Logging
    log_to_file: bool = True
    log_directory: Path = Path("data/sessions")
    
    # Limits
    max_moves: Optional[int] = None       # Stop after N moves (None = unlimited)


class AutoPlayController:
    """
    Controls game automatically by generating directions.
    
    The controller generates directions one at a time, only generating
    the next direction after the current move has completed. All
    generated directions are logged to a file.
    
    Usage:
        controller = AutoPlayController(config)
        controller.initialize(game_manager)
        controller.start()
        
        # In game loop:
        controller.update(delta_ms)
        
        # When done:
        controller.stop()
    """
    
    def __init__(self, config: AutoPlayConfig = None):
        self.config = config or AutoPlayConfig()
        
        # State
        self._active = False
        self._waiting_for_move = False
        self._delay_timer = 0.0
        self._move_count = 0
        self._last_direction: Optional[Direction] = None
        
        # Game reference
        self._game_manager = None
        self._is_walkable_func: Optional[Callable[[int, int], bool]] = None
        
        # Logging
        self._log_file: Optional[Path] = None
        self._log_handle = None
        
        # Random generator
        if self.config.random_seed is not None:
            self._rng = random.Random(self.config.random_seed)
        else:
            self._rng = random.Random()
            
        # Callbacks
        self._on_direction_generated: Optional[Callable[[Direction, int], None]] = None
        self._on_move_complete: Optional[Callable[[int], None]] = None
        
        # Direction opposites for backtrack avoidance
        self._opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        
    def initialize(self, game_manager):
        """
        Initialize the controller with a game manager.
        
        Args:
            game_manager: GameManager instance to control
        """
        self._game_manager = game_manager
        
        # Set up logging
        if self.config.log_to_file:
            self._setup_logging()
            
    def _setup_logging(self):
        """Set up the direction log file"""
        # Create log directory if needed
        log_dir = self.config.log_directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = log_dir / f"auto_play_{timestamp}.txt"
        
        # Open file and write header
        self._log_handle = open(self._log_file, 'w')
        self._log_handle.write("# Auto-Play Direction Log\n")
        self._log_handle.write(f"# Started: {datetime.now().isoformat()}\n")
        self._log_handle.write(f"# Seed: {self.config.random_seed}\n")
        self._log_handle.write("# Format: move_number,direction,timestamp,was_valid\n")
        self._log_handle.write("#\n")
        self._log_handle.flush()
        
        print(f"Auto-play logging to: {self._log_file}")
        
    def start(self):
        """Start auto-play mode"""
        if self._game_manager is None:
            raise RuntimeError("Controller not initialized. Call initialize() first.")
            
        self._active = True
        self._waiting_for_move = False
        self._delay_timer = 0.0
        self._move_count = 0
        self._last_direction = None
        
        print("Auto-play started")
        self._log_event("AUTO_PLAY_START")
        
    def stop(self):
        """Stop auto-play mode"""
        self._active = False
        self._waiting_for_move = False
        
        print(f"Auto-play stopped after {self._move_count} moves")
        self._log_event("AUTO_PLAY_STOP")
        
        # Close log file
        if self._log_handle:
            self._log_handle.write(f"# Ended: {datetime.now().isoformat()}\n")
            self._log_handle.write(f"# Total moves: {self._move_count}\n")
            self._log_handle.close()
            self._log_handle = None
            
    def update(self, delta_ms: float):
        """
        Update the controller. Call this every frame.
        
        Args:
            delta_ms: Time elapsed since last update in milliseconds
        """
        if not self._active:
            return
            
        # Check if we've reached max moves
        if self.config.max_moves and self._move_count >= self.config.max_moves:
            self.stop()
            return
            
        # If waiting for current move to complete, check status
        if self._waiting_for_move:
            if self._game_manager.can_accept_input:
                # Move completed, start delay
                self._waiting_for_move = False
                self._delay_timer = 0.0
                
                if self._on_move_complete:
                    self._on_move_complete(self._move_count)
            return
            
        # Apply delay between moves
        self._delay_timer += delta_ms
        if self._delay_timer < self.config.delay_between_moves_ms:
            return
            
        # Ready for next move - generate and execute
        if self._game_manager.can_accept_input:
            self._generate_and_execute_move()
            
    def _generate_and_execute_move(self):
        """Generate a direction and move the player"""
        # Get current player position
        player_pos = self._game_manager.player.grid_position
        maze = self._game_manager.maze
        
        # Generate direction
        direction = self._generate_direction(player_pos, maze)
        
        # Execute move
        moved = self._game_manager.move_player(direction)
        
        # Log the move
        self._log_direction(direction, moved)
        
        # Update state
        self._move_count += 1
        self._last_direction = direction
        self._waiting_for_move = True
        
        # Callback
        if self._on_direction_generated:
            self._on_direction_generated(direction, self._move_count)
            
        print(f"Auto-play [{self._move_count}]: {direction.value} {'(valid)' if moved else '(blocked)'}")
        
    def _generate_direction(self, player_pos: Tuple[int, int], maze) -> Direction:
        """
        Generate the next direction to move.
        
        Uses intelligent selection:
        1. If prefer_valid_moves: prioritize directions leading to walkable cells
        2. If avoid_backtracking: don't immediately reverse direction
        3. Random selection from remaining valid options
        
        Args:
            player_pos: Current (x, y) grid position
            maze: Maze object with is_walkable method
            
        Returns:
            Direction to move
        """
        all_directions = list(Direction.all())
        valid_directions = []
        
        # Find valid directions (ones that lead to walkable cells)
        x, y = player_pos
        for direction in all_directions:
            # Get target position
            dx, dy = self._get_direction_delta(direction)
            target_x, target_y = x + dx, y + dy
            
            # Check if walkable
            if maze.is_walkable(target_x, target_y):
                valid_directions.append(direction)
                
        # Apply backtrack avoidance
        if self.config.avoid_backtracking and self._last_direction:
            opposite = self._opposites.get(self._last_direction)
            if opposite in valid_directions and len(valid_directions) > 1:
                valid_directions.remove(opposite)
                
        # Choose from valid directions if available, otherwise any direction
        if self.config.prefer_valid_moves and valid_directions:
            return self._rng.choice(valid_directions)
        elif valid_directions:
            return self._rng.choice(valid_directions)
        else:
            # No valid moves - pick random (will be blocked)
            return self._rng.choice(all_directions)
            
    def _get_direction_delta(self, direction: Direction) -> Tuple[int, int]:
        """Get grid delta for a direction"""
        deltas = {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
        }
        return deltas[direction]
        
    def _log_direction(self, direction: Direction, was_valid: bool):
        """Log a generated direction to file"""
        if self._log_handle:
            timestamp = time.time()
            line = f"{self._move_count},{direction.value},{timestamp:.3f},{int(was_valid)}\n"
            self._log_handle.write(line)
            self._log_handle.flush()
            
    def _log_event(self, event: str):
        """Log an event to file"""
        if self._log_handle:
            timestamp = time.time()
            self._log_handle.write(f"# {event}: {timestamp:.3f}\n")
            self._log_handle.flush()
            
    def set_callbacks(
        self,
        on_direction_generated: Callable[[Direction, int], None] = None,
        on_move_complete: Callable[[int], None] = None,
    ):
        """Set callback functions"""
        self._on_direction_generated = on_direction_generated
        self._on_move_complete = on_move_complete
        
    @property
    def is_active(self) -> bool:
        """Whether auto-play is currently active"""
        return self._active
        
    @property
    def move_count(self) -> int:
        """Number of moves executed"""
        return self._move_count
        
    @property
    def log_file(self) -> Optional[Path]:
        """Path to current log file"""
        return self._log_file


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Simple test without pygame
    print("AutoPlayController module loaded successfully")
    
    # Test direction generation
    controller = AutoPlayController(AutoPlayConfig(
        random_seed=42,
        log_to_file=False,
    ))
    
    print("\nTest random directions (seed=42):")
    for i in range(10):
        # Simulate getting a direction
        direction = controller._rng.choice(list(Direction.all()))
        print(f"  {i+1}: {direction.value}")
