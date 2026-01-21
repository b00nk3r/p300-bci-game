"""
Maze Generation and Structure
=============================
Handles maze creation, storage, and pathfinding utilities.

Uses recursive backtracking algorithm for maze generation,
which creates perfect mazes (exactly one path between any two points).

The maze is stored as a 2D grid where each cell can be:
- WALL: Impassable
- PATH: Walkable
- START: Player starting position
- GOAL: Level goal (optional)
"""

import random
from enum import IntEnum
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass


class CellType(IntEnum):
    """Types of cells in the maze"""
    WALL = 0
    PATH = 1
    START = 2
    GOAL = 3


@dataclass
class MazeConfig:
    """Configuration for maze generation"""
    # Maze dimensions (in cells, must be odd for algorithm to work properly)
    width: int = 21          # Number of columns
    height: int = 15         # Number of rows
    
    # Cell size in pixels (for rendering)
    cell_size: int = 32
    
    # Generation settings
    seed: Optional[int] = None  # Random seed for reproducibility
    
    # Difficulty settings
    remove_dead_ends: float = 0.0  # 0.0-1.0, removes dead ends to create loops
    
    # Forbidden zone (area that must remain walls - e.g., arrow panel)
    # In grid coordinates
    forbidden_rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    
    # Generation mode
    use_corridors: bool = False  # If True, generate simple corridors instead of maze
    
    def __post_init__(self):
        # Ensure odd dimensions for proper maze generation (only if not using corridors)
        if not self.use_corridors:
            if self.width % 2 == 0:
                self.width += 1
            if self.height % 2 == 0:
                self.height += 1


class Maze:
    """
    Maze structure and generation.
    
    The maze uses a grid where:
    - Odd coordinates (1,1), (1,3), (3,1), etc. are potential paths
    - Even coordinates are walls or connections between paths
    - Forbidden zone cells always remain as walls
    
    Usage:
        config = MazeConfig(width=21, height=15)
        maze = Maze(config)
        maze.generate()
        
        # Access cells
        cell = maze.get_cell(x, y)
        
        # Get positions
        start = maze.start_pos
        goal = maze.goal_pos
    """
    
    def __init__(self, config: MazeConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        
        # Grid storage
        self._grid: List[List[CellType]] = []
        
        # Forbidden zone (rect in grid coords)
        self._forbidden_rect: Optional[Tuple[int, int, int, int]] = config.forbidden_rect
        
        # Plus-shaped forbidden zone (alternative to rectangle)
        self._forbidden_plus: Optional[dict] = None
        
        # Special positions
        self._start_pos: Tuple[int, int] = (1, 1)
        self._goal_pos: Tuple[int, int] = (config.width - 2, config.height - 2)
        
        # Initialize empty grid
        self._init_grid()
        
    def set_forbidden_zone(self, rect: Tuple[int, int, int, int]):
        """
        Set forbidden zone where paths cannot be carved.
        
        Args:
            rect: (x, y, width, height) in grid coordinates
        """
        self._forbidden_rect = rect
        self._forbidden_plus = None  # Clear plus shape if rectangle is set
    
    def set_forbidden_plus(self, vertical: Tuple[int, int, int, int], horizontal: Tuple[int, int, int, int]):
        """
        Set plus-shaped forbidden zone.
        
        Args:
            vertical: (x, y, width, height) of vertical strip in grid coordinates
            horizontal: (x, y, width, height) of horizontal strip in grid coordinates
        """
        self._forbidden_plus = {
            'vertical': vertical,
            'horizontal': horizontal,
        }
        self._forbidden_rect = None  # Clear rectangle if plus is set
        
    def is_forbidden(self, x: int, y: int) -> bool:
        """Check if cell is in forbidden zone (rectangle or plus shape)"""
        # Check plus-shaped forbidden zone
        if self._forbidden_plus is not None:
            # Check vertical strip
            vx, vy, vw, vh = self._forbidden_plus['vertical']
            if vx <= x < vx + vw and vy <= y < vy + vh:
                return True
            # Check horizontal strip
            hx, hy, hw, hh = self._forbidden_plus['horizontal']
            if hx <= x < hx + hw and hy <= y < hy + hh:
                return True
            return False
        
        # Check rectangular forbidden zone
        if self._forbidden_rect is None:
            return False
        fx, fy, fw, fh = self._forbidden_rect
        return fx <= x < fx + fw and fy <= y < fy + fh
        
    def _init_grid(self):
        """Initialize grid with all walls"""
        self._grid = [
            [CellType.WALL for _ in range(self.width)]
            for _ in range(self.height)
        ]
        
    def generate(self, seed: Optional[int] = None):
        """
        Generate a new maze.
        
        Uses either:
        - Recursive backtracking for complex mazes
        - Corridor generation for simple paths (when use_corridors=True)
        
        Args:
            seed: Random seed for reproducibility (overrides config)
        """
        # Set random seed
        if seed is not None:
            random.seed(seed)
        elif self.config.seed is not None:
            random.seed(self.config.seed)
            
        # Reset grid
        self._init_grid()
        
        if self.config.use_corridors:
            # Simple corridor generation for large cells
            self._generate_corridors()
        else:
            # Traditional maze generation
            # Start from (1, 1) - must be odd coordinates
            self._carve_passages(1, 1)
        
        # Set start and goal positions
        self._place_start_and_goal()
        
        # Optionally remove some dead ends to create loops (only for maze mode)
        if not self.config.use_corridors and self.config.remove_dead_ends > 0:
            self._remove_dead_ends()
    
    def _generate_corridors(self):
        """
        Generate simple corridors around the forbidden zone.
        
        Creates a continuous path system that:
        - Forms paths around the arrow panel (rectangular or plus-shaped)
        - Uses corner areas when available (plus shape)
        - Ensures full connectivity from any point to any other
        """
        # Start by making everything a path (except borders and forbidden)
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if not self.is_forbidden(x, y):
                    self._grid[y][x] = CellType.PATH
        
        # Add some strategic walls to make it more interesting
        # But be careful not to block connectivity
        
        # Only add internal walls if we have enough space
        # For plus-shaped zones, the corners are already open
        if self._forbidden_plus:
            # Plus-shaped: add some walls but keep corners fully open
            vx, vy, vw, vh = self._forbidden_plus['vertical']
            hx, hy, hw, hh = self._forbidden_plus['horizontal']
            
            # Add walls in non-corner areas only
            # Left side walls (between left edge and vertical strip, outside horizontal strip)
            for y in range(2, self.height - 2):
                if not (hy <= y < hy + hh):  # Not in horizontal strip row
                    if 2 < vx - 2:  # If there's room for walls
                        x = 3
                        if not self.is_forbidden(x, y) and y % 3 != 0:
                            self._grid[y][x] = CellType.WALL
            
            # Right side walls
            for y in range(2, self.height - 2):
                if not (hy <= y < hy + hh):  # Not in horizontal strip row
                    right_x = self.width - 4
                    if vx + vw + 2 < right_x:  # If there's room
                        if not self.is_forbidden(right_x, y) and y % 3 != 0:
                            self._grid[y][right_x] = CellType.WALL
        
        elif self._forbidden_rect:
            # Rectangular: original wall placement
            fx, fy, fw, fh = self._forbidden_rect
            
            if fx > 4:
                for y in range(3, self.height - 3):
                    if y % 3 != 0:
                        self._grid[y][3] = CellType.WALL
            
            if self.width - (fx + fw) > 4:
                right_wall_x = self.width - 4
                for y in range(3, self.height - 3):
                    if y % 3 != 0:
                        self._grid[y][right_wall_x] = CellType.WALL
    
    def _place_start_and_goal(self):
        """Place start and goal positions on valid path cells"""
        # Find valid start position (prefer top-left area)
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self._grid[y][x] == CellType.PATH and not self.is_forbidden(x, y):
                    self._start_pos = (x, y)
                    self._grid[y][x] = CellType.START
                    break
            else:
                continue
            break
        
        # Find valid goal position (prefer bottom-right area)
        for y in range(self.height - 2, 0, -1):
            for x in range(self.width - 2, 0, -1):
                if self._grid[y][x] == CellType.PATH and not self.is_forbidden(x, y):
                    self._goal_pos = (x, y)
                    self._grid[y][x] = CellType.GOAL
                    break
            else:
                continue
            break
            
    def _carve_passages(self, start_x: int, start_y: int):
        """
        Carve passages using iterative backtracking (avoids recursion limit).
        
        Uses an explicit stack instead of recursion to handle large mazes.
        
        Args:
            start_x, start_y: Starting cell position (must be odd coordinates)
        """
        # Use explicit stack instead of recursion
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack[-1]
            
            # Mark current cell as path if not already
            if self._grid[y][x] == CellType.WALL and not self.is_forbidden(x, y):
                self._grid[y][x] = CellType.PATH
            
            # Get unvisited neighbors
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:  # Up, Down, Left, Right
                nx, ny = x + dx, y + dy
                
                # Check if valid, unvisited, and not forbidden
                if (0 < nx < self.width - 1 and 
                    0 < ny < self.height - 1 and
                    self._grid[ny][nx] == CellType.WALL and
                    not self.is_forbidden(nx, ny) and
                    not self.is_forbidden(x + dx // 2, y + dy // 2)):
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Choose random neighbor
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Carve passage between current and neighbor
                self._grid[y + dy // 2][x + dx // 2] = CellType.PATH
                
                # Push neighbor onto stack
                stack.append((nx, ny))
            else:
                # No unvisited neighbors, backtrack
                stack.pop()
                
    def _remove_dead_ends(self):
        """Remove some dead ends to create loops (makes maze easier)"""
        dead_ends = self._find_dead_ends()
        
        num_to_remove = int(len(dead_ends) * self.config.remove_dead_ends)
        
        for _ in range(num_to_remove):
            if not dead_ends:
                break
                
            # Pick random dead end
            x, y = random.choice(list(dead_ends))
            dead_ends.remove((x, y))
            
            # Find a wall neighbor that could be opened
            neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            wall_neighbors = [
                (nx, ny) for nx, ny in neighbors
                if 0 < nx < self.width - 1 and 0 < ny < self.height - 1
                and self._grid[ny][nx] == CellType.WALL
            ]
            
            if wall_neighbors:
                wx, wy = random.choice(wall_neighbors)
                self._grid[wy][wx] = CellType.PATH
                
    def _find_dead_ends(self) -> Set[Tuple[int, int]]:
        """Find all dead end cells (path cells with only one path neighbor)"""
        dead_ends = set()
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self._grid[y][x] in (CellType.PATH, CellType.START, CellType.GOAL):
                    # Count path neighbors
                    path_neighbors = sum(
                        1 for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                        if self._grid[y + dy][x + dx] != CellType.WALL
                    )
                    if path_neighbors == 1:
                        dead_ends.add((x, y))
                        
        return dead_ends
        
    def get_cell(self, x: int, y: int) -> CellType:
        """Get cell type at position"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._grid[y][x]
        return CellType.WALL
        
    def set_cell(self, x: int, y: int, cell_type: CellType):
        """Set cell type at position"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self._grid[y][x] = cell_type
            
    def is_walkable(self, x: int, y: int) -> bool:
        """Check if cell is walkable (not a wall and not in forbidden zone)"""
        if self.is_forbidden(x, y):
            return False
        return self.get_cell(x, y) != CellType.WALL
        
    def get_walkable_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get walkable neighboring cells"""
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
        
    def get_random_path_cell(self, exclude: Set[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Get a random walkable cell, optionally excluding some positions"""
        exclude = exclude or set()
        
        candidates = []
        for y in range(self.height):
            for x in range(self.width):
                if ((x, y) not in exclude and 
                    not self.is_forbidden(x, y) and
                    self._grid[y][x] in (CellType.PATH,)):
                    candidates.append((x, y))
                    
        return random.choice(candidates) if candidates else None
        
    @property
    def start_pos(self) -> Tuple[int, int]:
        """Get start position"""
        return self._start_pos
        
    @property
    def goal_pos(self) -> Tuple[int, int]:
        """Get goal position"""
        return self._goal_pos
        
    @property
    def grid(self) -> List[List[CellType]]:
        """Get the grid (read-only access)"""
        return self._grid
        
    def to_string(self) -> str:
        """Convert maze to string representation for debugging"""
        chars = {
            CellType.WALL: '█',
            CellType.PATH: ' ',
            CellType.START: 'S',
            CellType.GOAL: 'G',
        }
        
        lines = []
        for row in self._grid:
            line = ''.join(chars.get(cell, '?') for cell in row)
            lines.append(line)
            
        return '\n'.join(lines)


# =============================================================================
# Testing
# =============================================================================

def demo():
    """Demo maze generation"""
    config = MazeConfig(width=31, height=15, seed=42)
    maze = Maze(config)
    maze.generate()
    
    print("Generated Maze:")
    print(maze.to_string())
    print()
    print(f"Start: {maze.start_pos}")
    print(f"Goal: {maze.goal_pos}")
    print(f"Size: {maze.width}x{maze.height}")
    

if __name__ == "__main__":
    demo()