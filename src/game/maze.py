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

    def is_forbidden(self, x: int, y: int) -> bool:
        """Check if cell is in the forbidden zone"""
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

    def _generate_corridors(self):
        """
        Generate open playing field.

        Creates a fully open play area where:
        - All cells are walkable paths (except forbidden zone)
        - No border walls - play field extends to screen edges
        - Player can move freely anywhere on the field
        """
        # Make everything a path (including borders, except forbidden zone)
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_forbidden(x, y):
                    self._grid[y][x] = CellType.PATH

        # No internal walls - completely open field

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