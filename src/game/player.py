"""
Player Entity
=============
Handles player state, movement, and animation.

Features:
- Grid-based movement
- Smooth animation between cells
- Movement validation
- Direction tracking
"""

import time
from enum import Enum, auto
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

from config import Direction


class PlayerState(Enum):
    """Player state"""
    IDLE = auto()
    MOVING = auto()


@dataclass
class PlayerConfig:
    """Player configuration"""
    # Movement
    move_duration_ms: float = 200.0  # Time to move one cell
    
    # Appearance (used by renderer)
    size_ratio: float = 0.7  # Size relative to cell size
    
    # Colors (simple shapes for now)
    color: Tuple[int, int, int] = (100, 180, 100)
    outline_color: Tuple[int, int, int] = (60, 140, 60)


class Player:
    """
    Player entity with grid-based movement.
    
    The player moves between cells with smooth animation.
    Position is tracked in grid coordinates, with interpolation
    for smooth visual movement.
    
    Usage:
        player = Player(config, start_pos=(1, 1))
        
        # Attempt move
        if player.can_move(Direction.UP, maze):
            player.move(Direction.UP)
            
        # Update animation
        player.update(delta_ms)
        
        # Get render position (interpolated)
        x, y = player.render_position
    """
    
    def __init__(
        self, 
        config: PlayerConfig,
        start_pos: Tuple[int, int] = (1, 1)
    ):
        self.config = config
        
        # Grid position (integer cell coordinates)
        self._grid_x: int = start_pos[0]
        self._grid_y: int = start_pos[1]
        
        # Previous position (for interpolation)
        self._prev_x: int = start_pos[0]
        self._prev_y: int = start_pos[1]
        
        # Movement state
        self._state = PlayerState.IDLE
        self._move_progress: float = 0.0  # 0.0 to 1.0
        self._current_direction: Optional[Direction] = None
        
        # Callbacks
        self._on_move_complete: Optional[Callable[[Tuple[int, int]], None]] = None
        
    def set_position(self, x: int, y: int):
        """Set player position directly (no animation)"""
        self._grid_x = x
        self._grid_y = y
        self._prev_x = x
        self._prev_y = y
        self._state = PlayerState.IDLE
        self._move_progress = 0.0
        
    def can_move(self, direction: Direction, is_walkable_func: Callable[[int, int], bool]) -> bool:
        """
        Check if player can move in direction.
        
        Args:
            direction: Direction to move
            is_walkable_func: Function that returns True if cell is walkable
            
        Returns:
            True if move is valid
        """
        if self._state == PlayerState.MOVING:
            return False
            
        dx, dy = self._direction_to_delta(direction)
        new_x = self._grid_x + dx
        new_y = self._grid_y + dy
        
        return is_walkable_func(new_x, new_y)
        
    def move(self, direction: Direction) -> bool:
        """
        Start moving in direction.
        
        Args:
            direction: Direction to move
            
        Returns:
            True if move started
        """
        if self._state == PlayerState.MOVING:
            return False
            
        # Store previous position
        self._prev_x = self._grid_x
        self._prev_y = self._grid_y
        
        # Calculate new position
        dx, dy = self._direction_to_delta(direction)
        self._grid_x += dx
        self._grid_y += dy
        
        # Start movement
        self._state = PlayerState.MOVING
        self._move_progress = 0.0
        self._current_direction = direction
        
        return True
        
    def update(self, delta_ms: float):
        """
        Update player state.
        
        Args:
            delta_ms: Time elapsed since last update in milliseconds
        """
        if self._state == PlayerState.MOVING:
            # Update movement progress
            self._move_progress += delta_ms / self.config.move_duration_ms
            
            if self._move_progress >= 1.0:
                # Movement complete
                self._move_progress = 1.0
                self._state = PlayerState.IDLE
                self._prev_x = self._grid_x
                self._prev_y = self._grid_y
                
                # Notify callback
                if self._on_move_complete:
                    self._on_move_complete((self._grid_x, self._grid_y))
                    
    def _direction_to_delta(self, direction: Direction) -> Tuple[int, int]:
        """Convert direction to grid delta"""
        deltas = {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
        }
        return deltas.get(direction, (0, 0))
        
    @property
    def grid_position(self) -> Tuple[int, int]:
        """Get current grid position (target if moving)"""
        return (self._grid_x, self._grid_y)
        
    @property
    def render_position(self) -> Tuple[float, float]:
        """
        Get interpolated position for rendering.
        
        Returns:
            (x, y) in grid coordinates (can be fractional during movement)
        """
        if self._state == PlayerState.IDLE:
            return (float(self._grid_x), float(self._grid_y))
            
        # Interpolate between previous and current position
        # Use smooth easing for nicer movement
        t = self._ease_out_quad(self._move_progress)
        
        x = self._prev_x + (self._grid_x - self._prev_x) * t
        y = self._prev_y + (self._grid_y - self._prev_y) * t
        
        return (x, y)
        
    def _ease_out_quad(self, t: float) -> float:
        """Quadratic ease-out for smooth deceleration"""
        return 1 - (1 - t) ** 2
        
    @property
    def is_moving(self) -> bool:
        """Whether player is currently moving"""
        return self._state == PlayerState.MOVING
        
    @property
    def current_direction(self) -> Optional[Direction]:
        """Current movement direction (None if idle)"""
        return self._current_direction if self._state == PlayerState.MOVING else None
        
    def set_move_callback(self, callback: Callable[[Tuple[int, int]], None]):
        """Set callback for when movement completes"""
        self._on_move_complete = callback


# =============================================================================
# Testing
# =============================================================================

def demo():
    """Demo player movement"""
    config = PlayerConfig()
    player = Player(config, start_pos=(1, 1))
    
    print("Player Demo")
    print(f"  Start: {player.grid_position}")
    
    # Simulate move
    def is_walkable(x, y):
        return True  # All cells walkable for demo
        
    if player.can_move(Direction.RIGHT, is_walkable):
        player.move(Direction.RIGHT)
        print(f"  Moving RIGHT to {player.grid_position}")
        
        # Simulate updates
        for i in range(10):
            player.update(25)  # 25ms per update
            rx, ry = player.render_position
            print(f"    t={i*25}ms: render=({rx:.2f}, {ry:.2f}) moving={player.is_moving}")
            
    print(f"  Final: {player.grid_position}")
    

if __name__ == "__main__":
    demo()
