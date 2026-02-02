"""
Game Renderer (Pixel Art Style)
================================
Renders game elements in a retro pixel art style.

Key design principles:
- Blocky, pixelated graphics for maze, player, and collectibles
- The plus-shaped arrow panel area remains UNCHANGED
- All colors remain grayscale as specified
- All sizes and proportions are preserved

Pixel art techniques used:
- Chunky grid-aligned shapes
- Sharp edges without anti-aliasing
- Simple iconic representations
- Highlight/shadow details for depth

Currently renders:
- Maze walls and paths with brick/tile textures
- Player as a pixel art character
- Collectibles as pixel art icons
- UI elements (score, status)
"""

import pygame
import math
import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from src.game.maze import Maze, CellType
from src.game.player import Player
from src.game.collectible import CollectibleManager, Collectible, CollectibleType


@dataclass
class RenderConfig:
    """Rendering configuration"""
    # Cell size
    cell_size: int = 32
    
    # Maze colors (much duller grayscale to make arrows stand out)
    wall_color: Tuple[int, int, int] = (30, 30, 30)
    path_color: Tuple[int, int, int] = (18, 18, 18)
    start_color: Tuple[int, int, int] = (25, 25, 25)
    goal_color: Tuple[int, int, int] = (32, 32, 32)
    
    # Player colors (much duller grayscale)
    player_color: Tuple[int, int, int] = (70, 70, 70)
    player_outline: Tuple[int, int, int] = (50, 50, 50)
    player_direction_color: Tuple[int, int, int] = (80, 80, 80)
    
    # Collectible colors (much duller gray shades)
    coin_color: Tuple[int, int, int] = (75, 75, 75)
    gem_color: Tuple[int, int, int] = (60, 60, 60)
    star_color: Tuple[int, int, int] = (80, 80, 80)
    
    # UI colors (much duller grayscale)
    ui_bg_color: Tuple[int, int, int] = (15, 15, 15)
    ui_text_color: Tuple[int, int, int] = (70, 70, 70)
    ui_highlight_color: Tuple[int, int, int] = (90, 90, 90)
    
    # Sizes
    player_size_ratio: float = 1.0   # Player fills entire cell
    collectible_size_ratio: float = 0.4  # Collectible size relative to cell


class GameRenderer:
    """
    Renders all game elements.
    
    Supports a "hole" in the middle of the maze where the arrow panel sits.
    The maze wraps around this hole.
    
    Usage:
        renderer = GameRenderer(config)
        renderer.setup(screen_width, screen_height, maze, arrow_panel_rect)
        
        # In game loop:
        renderer.draw_maze(screen, maze)
        renderer.draw_collectibles(screen, collectible_manager)
        renderer.draw_player(screen, player)
        renderer.draw_ui(screen, score, level)
    """
    
    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        
        # Calculated values (set in setup)
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._maze_width_px: int = 0
        self._maze_height_px: int = 0
        
        # Arrow panel hole (in grid coordinates)
        self._hole_rect: Optional[pygame.Rect] = None  # Screen coords
        self._hole_grid_rect: Optional[pygame.Rect] = None  # Grid coords
        
        # Cached surfaces for performance
        self._maze_surface: Optional[pygame.Surface] = None
        self._maze_dirty: bool = True
        
        # Pixel art texture caches
        self._wall_texture: Optional[pygame.Surface] = None
        self._path_texture: Optional[pygame.Surface] = None
        self._start_texture: Optional[pygame.Surface] = None
        self._goal_texture: Optional[pygame.Surface] = None
        self._player_sprites: Dict[str, pygame.Surface] = {}
        self._collectible_sprites: Dict[str, pygame.Surface] = {}
        
        # Fonts
        self._font_large: Optional[pygame.font.Font] = None
        self._font_medium: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        
        # Animation time
        self._start_time: float = time.time()
        
    def setup(
        self, 
        screen_width: int, 
        screen_height: int, 
        maze: Maze,
        arrow_panel_rect: pygame.Rect = None
    ):
        """
        Setup renderer for given screen and maze.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            maze: Maze to render
            arrow_panel_rect: Rectangle of arrow panel (for reference)
        """
        cell_size = self.config.cell_size
        
        # Calculate maze dimensions
        self._maze_width_px = maze.width * cell_size
        self._maze_height_px = maze.height * cell_size
        
        # Center maze on screen
        self._offset_x = (screen_width - self._maze_width_px) // 2
        self._offset_y = (screen_height - self._maze_height_px) // 2
        
        # Store maze reference for scoreboard zone
        self._maze_width = maze.width
        self._maze_height = maze.height
        
        # Scoreboard position: 2 cells in top-right corner
        self._scoreboard_cells = [
            (maze.width - 2, 0),  # Left cell of scoreboard
            (maze.width - 1, 0),  # Right cell of scoreboard
        ]
        
        # The hole is now determined by the maze's forbidden zone
        # which can be either rectangular or plus-shaped
        self._hole_rect = arrow_panel_rect
        
        # For rendering, we use the maze's is_forbidden() method
        # which handles both rectangular and plus-shaped zones
        if maze._forbidden_plus:
            # Store plus shape for hole rendering
            vx, vy, vw, vh = maze._forbidden_plus['vertical']
            hx, hy, hw, hh = maze._forbidden_plus['horizontal']
            self._hole_grid_rect = None  # Not using single rect
            self._hole_plus = {
                'vertical': pygame.Rect(vx, vy, vw, vh),
                'horizontal': pygame.Rect(hx, hy, hw, hh),
            }
        elif maze._forbidden_rect:
            fx, fy, fw, fh = maze._forbidden_rect
            self._hole_grid_rect = pygame.Rect(fx, fy, fw, fh)
            self._hole_plus = None
        else:
            self._hole_grid_rect = None
            self._hole_plus = None
        
        # Initialize fonts
        self._font_large = pygame.font.Font(None, 36)
        self._font_medium = pygame.font.Font(None, 28)
        self._font_small = pygame.font.Font(None, 22)
        
        # Create pixel art textures for this cell size
        self._create_pixel_textures(cell_size)
        
        # Create scoreboard texture
        self._create_scoreboard_texture(cell_size)
        
        # Create maze surface cache
        self._maze_surface = pygame.Surface(
            (self._maze_width_px, self._maze_height_px),
            pygame.SRCALPHA  # Support transparency for hole
        )
        self._maze_dirty = True
    
    def _create_pixel_textures(self, cell_size: int):
        """Create pixel art textures for maze tiles"""
        # Wall texture - brick pattern
        self._wall_texture = self._create_brick_texture(
            cell_size, self.config.wall_color
        )
        
        # Path textures - two versions for alternating pattern
        # Version A: starts with light tile in top-left
        # Version B: starts with dark tile in top-left
        self._path_texture_a = self._create_floor_texture(
            cell_size, self.config.path_color, start_light=True
        )
        self._path_texture_b = self._create_floor_texture(
            cell_size, self.config.path_color, start_light=False
        )
        
        # Start textures - two versions for alternating pattern
        self._start_texture_a = self._create_start_texture(
            cell_size, self.config.start_color, start_light=True
        )
        self._start_texture_b = self._create_start_texture(
            cell_size, self.config.start_color, start_light=False
        )
        
        # Goal textures - two versions for alternating pattern
        self._goal_texture_a = self._create_goal_texture(
            cell_size, self.config.goal_color, start_light=True
        )
        self._goal_texture_b = self._create_goal_texture(
            cell_size, self.config.goal_color, start_light=False
        )
        
        # Create player sprites
        self._create_player_sprites(cell_size)
        
        # Create collectible sprites
        self._create_collectible_sprites(cell_size)
    
    def _create_brick_texture(self, size: int, base_color: Tuple[int, int, int]) -> pygame.Surface:
        """Create a true pixel art brick wall texture with small visible pixels"""
        surface = pygame.Surface((size, size))
        
        # Small pixel size for authentic pixel art look
        pixel_size = max(2, size // 40)  # ~4px pixels for 160px cell
        
        # Color palette
        mortar = tuple(max(0, c - 35) for c in base_color)
        mortar_dark = tuple(max(0, c - 45) for c in base_color)
        brick_base = base_color
        brick_dark = tuple(max(0, c - 12) for c in base_color)
        brick_darker = tuple(max(0, c - 22) for c in base_color)
        brick_light = tuple(min(255, c + 15) for c in base_color)
        brick_lighter = tuple(min(255, c + 28) for c in base_color)
        highlight = tuple(min(255, c + 40) for c in base_color)
        
        # Fill with mortar base
        surface.fill(mortar)
        
        # Target larger bricks (~53x26 for 160px cell = 3 wide x 6 tall)
        target_brick_width = 53
        target_brick_height = 26
        mortar_thickness = pixel_size
        
        bricks_per_row = max(1, round(size / target_brick_width))
        brick_width = size // bricks_per_row
        
        rows_per_cell = max(1, round(size / target_brick_height))
        brick_height = size // rows_per_cell
        
        # Pixels per brick
        pixels_wide = brick_width // pixel_size
        pixels_tall = brick_height // pixel_size
        
        for row in range(rows_per_cell + 1):
            base_y = row * brick_height
            # Offset every other row
            offset = (brick_width // 2) if row % 2 else 0
            
            for col in range(-1, bricks_per_row + 2):
                base_x = col * brick_width + offset
                
                # Skip if brick is completely outside
                if base_x + brick_width <= 0 or base_x >= size:
                    continue
                if base_y + brick_height <= 0 or base_y >= size:
                    continue
                
                # Draw each pixel within the brick
                for py in range(pixels_tall):
                    for px in range(pixels_wide):
                        x = base_x + px * pixel_size
                        y = base_y + py * pixel_size
                        
                        # Skip if outside surface
                        if x < 0 or x >= size or y < 0 or y >= size:
                            continue
                        
                        # Check if this pixel is in the mortar area
                        is_mortar_bottom = py >= pixels_tall - 1
                        is_mortar_right = px >= pixels_wide - 1
                        
                        if is_mortar_bottom or is_mortar_right:
                            # Mortar pixels
                            if is_mortar_bottom and is_mortar_right:
                                color = mortar_dark
                            else:
                                color = mortar
                        else:
                            # Brick pixels
                            is_top_edge = py == 0
                            is_left_edge = px == 0
                            is_near_bottom = py == pixels_tall - 2
                            is_near_right = px == pixels_wide - 2
                            is_near_top = py == 1
                            is_near_left = px == 1
                            
                            if is_top_edge and is_left_edge:
                                color = highlight  # Corner highlight
                            elif is_top_edge:
                                color = brick_lighter  # Top edge
                            elif is_left_edge:
                                color = brick_light  # Left edge
                            elif is_near_bottom or is_near_right:
                                color = brick_darker  # Near shadow edge
                            elif is_near_top:
                                color = brick_light  # Inner top highlight
                            elif is_near_left:
                                color = brick_light  # Inner left highlight
                            else:
                                # Interior - subtle texture
                                noise = ((px * 7 + py * 13 + row * 3 + col * 5) % 7)
                                if noise == 0:
                                    color = brick_light
                                elif noise == 1:
                                    color = brick_dark
                                else:
                                    color = brick_base
                            
                        pygame.draw.rect(surface, color, (x, y, pixel_size, pixel_size))
        
        return surface
    
    def _create_floor_texture(self, size: int, base_color: Tuple[int, int, int], start_light: bool = True) -> pygame.Surface:
        """Create a true pixel art floor tile texture with small visible pixels
        
        Args:
            size: Texture size in pixels
            base_color: Base color for the tiles
            start_light: If True, top-left tile is light. If False, top-left tile is dark.
        """
        surface = pygame.Surface((size, size))
        
        # Small pixel size for authentic pixel art look (like the Viking sprite)
        pixel_size = max(2, size // 40)  # ~4px pixels for 160px cell
        
        # Color palette with more variation for pixel art depth
        base = base_color
        dark = tuple(max(0, c - 15) for c in base_color)
        darker = tuple(max(0, c - 25) for c in base_color)
        light = tuple(min(255, c + 12) for c in base_color)
        lighter = tuple(min(255, c + 22) for c in base_color)
        highlight = tuple(min(255, c + 35) for c in base_color)
        groove = tuple(max(0, c - 35) for c in base_color)
        
        # 3x3 tiles per cell (larger tiles)
        tiles_per_side = 3
        tile_size = size // tiles_per_side
        
        # Pixels per tile
        pixels_per_tile = tile_size // pixel_size
        
        for tile_row in range(tiles_per_side):
            for tile_col in range(tiles_per_side):
                tile_x = tile_col * tile_size
                tile_y = tile_row * tile_size
                
                # Determine if this is a light or dark tile
                # Base pattern: (tile_row + tile_col) % 2 == 0 means light
                # If start_light is False, invert the pattern
                base_is_light = (tile_row + tile_col) % 2 == 0
                is_light_tile = base_is_light if start_light else not base_is_light
                
                # Draw each pixel within the tile
                for py in range(pixels_per_tile):
                    for px in range(pixels_per_tile):
                        x = tile_x + px * pixel_size
                        y = tile_y + py * pixel_size
                        
                        # Determine pixel color based on position within tile
                        # This creates the pixel art texture effect
                        
                        # Edge detection
                        is_top_edge = py == 0
                        is_left_edge = px == 0
                        is_bottom_edge = py == pixels_per_tile - 1
                        is_right_edge = px == pixels_per_tile - 1
                        is_corner = (is_top_edge or is_bottom_edge) and (is_left_edge or is_right_edge)
                        
                        # Near-edge (1 pixel in from edge)
                        is_near_top = py == 1
                        is_near_left = px == 1
                        is_near_bottom = py == pixels_per_tile - 2
                        is_near_right = px == pixels_per_tile - 2
                        
                        if is_light_tile:
                            # Light tile coloring
                            if is_top_edge and is_left_edge:
                                color = highlight  # Top-left corner highlight
                            elif is_top_edge or is_left_edge:
                                color = lighter  # Top and left edges
                            elif is_bottom_edge or is_right_edge:
                                color = groove  # Bottom and right grooves
                            elif is_near_top or is_near_left:
                                color = light  # Inner highlight
                            elif is_near_bottom or is_near_right:
                                color = dark  # Inner shadow
                            else:
                                # Interior pixels - add subtle noise pattern
                                if (px + py) % 3 == 0:
                                    color = light
                                elif (px + py) % 5 == 0:
                                    color = dark
                                else:
                                    color = base
                        else:
                            # Dark tile coloring
                            if is_top_edge and is_left_edge:
                                color = light  # Dimmer highlight
                            elif is_top_edge or is_left_edge:
                                color = base  # Edges
                            elif is_bottom_edge or is_right_edge:
                                color = groove  # Grooves
                            elif is_near_top or is_near_left:
                                color = dark
                            elif is_near_bottom or is_near_right:
                                color = darker
                            else:
                                # Interior pixels - add subtle noise pattern
                                if (px + py) % 3 == 0:
                                    color = base
                                elif (px + py) % 5 == 0:
                                    color = darker
                                else:
                                    color = dark
                        
                        pygame.draw.rect(surface, color, (x, y, pixel_size, pixel_size))
        
        return surface
    
    def _create_start_texture(self, size: int, base_color: Tuple[int, int, int], start_light: bool = True) -> pygame.Surface:
        """Create a pixel art start position texture (same as floor, no markings)"""
        return self._create_floor_texture(size, base_color, start_light=start_light)
    
    def _create_goal_texture(self, size: int, base_color: Tuple[int, int, int], start_light: bool = True) -> pygame.Surface:
        """Create a pixel art goal position texture (same as floor, no markings)"""
        return self._create_floor_texture(size, base_color, start_light=start_light)
    
    def _create_scoreboard_texture(self, cell_size: int):
        """Load pixel art scoreboard image spanning 2 cells.
        
        The board is in the top-right corner, so we view it from the left.
        Uses the pre-made scoreboard.png image.
        """
        width = cell_size * 2
        height = cell_size
        
        pixel_size = max(2, cell_size // 40)  # Match floor tile pixel size
        
        # Try to load the scoreboard image
        scoreboard_path = None
        possible_paths = [
            "assets/scoreboard.png",
            "../assets/scoreboard.png",
            "src/game/../../assets/scoreboard.png",
        ]
        
        import os
        for path in possible_paths:
            if os.path.exists(path):
                scoreboard_path = path
                break
        
        # Also try relative to this file's location
        if scoreboard_path is None:
            import pathlib
            this_dir = pathlib.Path(__file__).parent.parent.parent
            asset_path = this_dir / "assets" / "scoreboard.png"
            if asset_path.exists():
                scoreboard_path = str(asset_path)
        
        if scoreboard_path and os.path.exists(scoreboard_path):
            # Load the image
            original = pygame.image.load(scoreboard_path).convert_alpha()
            
            # The image has a checkered pattern for transparency - convert it
            # Make pixels that are light gray (the checkered background) transparent
            orig_width, orig_height = original.get_size()
            
            # First pass: convert checkered to transparent and find content bounds
            min_x, min_y = orig_width, orig_height
            max_x, max_y = 0, 0
            
            for y in range(orig_height):
                for x in range(orig_width):
                    r, g, b, a = original.get_at((x, y))
                    # Check if this is a light pixel (checkered background)
                    if r > 180 and g > 180 and b > 180:
                        original.set_at((x, y), (0, 0, 0, 0))
                    else:
                        # Track content bounds
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
            
            # Crop to content bounds with small padding
            padding = 2
            crop_x = max(0, min_x - padding)
            crop_y = max(0, min_y - padding)
            crop_w = min(orig_width - crop_x, max_x - min_x + padding * 2)
            crop_h = min(orig_height - crop_y, max_y - min_y + padding * 2)
            
            # Create cropped surface
            cropped = pygame.Surface((crop_w, crop_h), pygame.SRCALPHA)
            cropped.blit(original, (0, 0), (crop_x, crop_y, crop_w, crop_h))
            
            # Scale cropped image to fill the space
            scale = min(width / crop_w, height / crop_h)
            new_width = int(crop_w * scale)
            new_height = int(crop_h * scale)
            
            scaled = pygame.transform.smoothscale(cropped, (new_width, new_height))
            
            # Create final surface and position at bottom (legs should touch bottom)
            self._scoreboard_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            self._scoreboard_surface.fill((0, 0, 0, 0))
            
            x_offset = (width - new_width) // 2
            y_offset = height - new_height  # Align to bottom
            self._scoreboard_surface.blit(scaled, (x_offset, y_offset))
            
            # Store board dimensions for text positioning
            # Move text more to the right to avoid frame border
            self._board_text_left = x_offset + int(new_width * 0.12)
            self._board_text_top = y_offset + int(new_height * 0.18)
            self._board_text_right = x_offset + int(new_width * 0.92)
            self._board_text_bottom = y_offset + int(new_height * 0.75)
            
            # Use medium pixel size for scoreboard text (3/4 of normal size)
            self._scoreboard_pixel_size = max(2, (pixel_size * 3) // 4)
            
            print(f"Loaded scoreboard from {scoreboard_path}, cropped to {crop_w}x{crop_h}, scaled to {new_width}x{new_height}")
        else:
            # Fallback: create a simple dark rectangle
            self._scoreboard_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            self._scoreboard_surface.fill((20, 20, 20, 240))
            pygame.draw.rect(self._scoreboard_surface, (50, 50, 50), (0, 0, width, height), 3)
            
            self._board_text_left = pixel_size * 4
            self._board_text_top = pixel_size * 4
            self._board_text_right = width - pixel_size * 4
            
            print("Scoreboard image not found, using fallback")
        
        # Create pixel art font for scoreboard
        self._create_pixel_font(pixel_size)
    
    def _create_pixel_font(self, pixel_size: int):
        """Create pixel art characters for scoreboard text"""
        # Store pixel size for drawing
        self._pixel_font_size = pixel_size
        
        # Chalk color (white/light gray)
        chalk = (200, 200, 195)
        
        # Define 5x7 pixel font patterns (each char is list of strings)
        self._pixel_chars = {
            '0': [
                " ### ",
                "#   #",
                "#   #",
                "#   #",
                "#   #",
                "#   #",
                " ### ",
            ],
            '1': [
                "  #  ",
                " ##  ",
                "  #  ",
                "  #  ",
                "  #  ",
                "  #  ",
                " ### ",
            ],
            '2': [
                " ### ",
                "#   #",
                "    #",
                "  ## ",
                " #   ",
                "#    ",
                "#####",
            ],
            '3': [
                " ### ",
                "#   #",
                "    #",
                "  ## ",
                "    #",
                "#   #",
                " ### ",
            ],
            '4': [
                "#   #",
                "#   #",
                "#   #",
                "#####",
                "    #",
                "    #",
                "    #",
            ],
            '5': [
                "#####",
                "#    ",
                "#### ",
                "    #",
                "    #",
                "#   #",
                " ### ",
            ],
            '6': [
                " ### ",
                "#    ",
                "#    ",
                "#### ",
                "#   #",
                "#   #",
                " ### ",
            ],
            '7': [
                "#####",
                "    #",
                "   # ",
                "  #  ",
                "  #  ",
                "  #  ",
                "  #  ",
            ],
            '8': [
                " ### ",
                "#   #",
                "#   #",
                " ### ",
                "#   #",
                "#   #",
                " ### ",
            ],
            '9': [
                " ### ",
                "#   #",
                "#   #",
                " ####",
                "    #",
                "    #",
                " ### ",
            ],
            'S': [
                " ### ",
                "#   #",
                "#    ",
                " ### ",
                "    #",
                "#   #",
                " ### ",
            ],
            'C': [
                " ### ",
                "#   #",
                "#    ",
                "#    ",
                "#    ",
                "#   #",
                " ### ",
            ],
            'O': [
                " ### ",
                "#   #",
                "#   #",
                "#   #",
                "#   #",
                "#   #",
                " ### ",
            ],
            'R': [
                "#### ",
                "#   #",
                "#   #",
                "#### ",
                "#  # ",
                "#   #",
                "#   #",
            ],
            'E': [
                "#####",
                "#    ",
                "#    ",
                "#### ",
                "#    ",
                "#    ",
                "#####",
            ],
            'L': [
                "#    ",
                "#    ",
                "#    ",
                "#    ",
                "#    ",
                "#    ",
                "#####",
            ],
            'V': [
                "#   #",
                "#   #",
                "#   #",
                "#   #",
                " # # ",
                " # # ",
                "  #  ",
            ],
            'I': [
                " ### ",
                "  #  ",
                "  #  ",
                "  #  ",
                "  #  ",
                "  #  ",
                " ### ",
            ],
            'T': [
                "#####",
                "  #  ",
                "  #  ",
                "  #  ",
                "  #  ",
                "  #  ",
                "  #  ",
            ],
            'M': [
                "#   #",
                "## ##",
                "# # #",
                "#   #",
                "#   #",
                "#   #",
                "#   #",
            ],
            ':': [
                "     ",
                "  #  ",
                "  #  ",
                "     ",
                "  #  ",
                "  #  ",
                "     ",
            ],
            '/': [
                "    #",
                "   # ",
                "   # ",
                "  #  ",
                " #   ",
                " #   ",
                "#    ",
            ],
            ' ': [
                "     ",
                "     ",
                "     ",
                "     ",
                "     ",
                "     ",
                "     ",
            ],
        }
    
    def _draw_pixel_text(self, surface: pygame.Surface, text: str, x: int, y: int, color: Tuple[int, int, int] = None):
        """Draw text using pixel art font"""
        if color is None:
            color = (200, 200, 195)  # Chalk white
        
        pixel_size = self._pixel_font_size
        char_width = 5 * pixel_size
        char_height = 7 * pixel_size
        spacing = pixel_size  # Space between characters
        
        cursor_x = x
        for char in text.upper():
            pattern = self._pixel_chars.get(char, self._pixel_chars.get(' '))
            if pattern:
                for row_idx, row in enumerate(pattern):
                    for col_idx, pixel in enumerate(row):
                        if pixel == '#':
                            px = cursor_x + col_idx * pixel_size
                            py = y + row_idx * pixel_size
                            pygame.draw.rect(surface, color, (px, py, pixel_size, pixel_size))
            cursor_x += char_width + spacing
    
    def _create_player_sprites(self, cell_size: int):
        """Load the Viking player sprite from image file"""
        size = int(cell_size * self.config.player_size_ratio)
        
        # Try to load the sprite image
        sprite_path = None
        possible_paths = [
            "assets/viking_transparent.png",
            "../assets/viking_transparent.png",
            "src/game/../../assets/viking_transparent.png",
        ]
        
        # Find the assets directory relative to working directory
        import os
        for path in possible_paths:
            if os.path.exists(path):
                sprite_path = path
                break
        
        # Also try relative to this file's location
        if sprite_path is None:
            import pathlib
            this_dir = pathlib.Path(__file__).parent.parent.parent
            asset_path = this_dir / "assets" / "viking_transparent.png"
            if asset_path.exists():
                sprite_path = str(asset_path)
        
        if sprite_path and os.path.exists(sprite_path):
            # Load the sprite image
            original_sprite = pygame.image.load(sprite_path).convert_alpha()
            
            # Scale to fit the cell size while maintaining aspect ratio
            orig_width, orig_height = original_sprite.get_size()
            
            # Scale to fill the cell (use the larger dimension to fill)
            scale = size / max(orig_width, orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Scale the sprite
            scaled_sprite = pygame.transform.smoothscale(original_sprite, (new_width, new_height))
            
            # Create a surface of the exact size needed, centered
            final_sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            final_sprite.fill((0, 0, 0, 0))
            
            # Center the scaled sprite
            x_offset = (size - new_width) // 2
            y_offset = (size - new_height) // 2
            final_sprite.blit(scaled_sprite, (x_offset, y_offset))
            
            # Use the same sprite for all directions (the character looks good facing forward)
            directions = ['up', 'down', 'left', 'right', 'idle']
            for direction in directions:
                # For left/right, we could flip the sprite, but the Viking looks symmetric
                if direction == 'left':
                    # Flip horizontally for left direction
                    flipped = pygame.transform.flip(final_sprite, True, False)
                    self._player_sprites[direction] = flipped
                else:
                    self._player_sprites[direction] = final_sprite.copy()
            
            print(f"Loaded Viking sprite from {sprite_path}, scaled to {size}x{size}")
        else:
            # Fallback: create a simple placeholder
            print(f"Warning: Could not find Viking sprite, using fallback")
            self._create_fallback_player_sprites(cell_size)
    
    def _create_fallback_player_sprites(self, cell_size: int):
        """Create simple fallback sprites if image not found"""
        size = int(cell_size * self.config.player_size_ratio)
        pixel_size = max(1, size // 16)
        
        color = self.config.player_color
        outline = self.config.player_outline
        
        directions = ['up', 'down', 'left', 'right', 'idle']
        
        for direction in directions:
            sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            sprite.fill((0, 0, 0, 0))
            
            # Simple circle as fallback
            center = size // 2
            radius = size // 2 - pixel_size * 2
            pygame.draw.circle(sprite, color, (center, center), radius)
            pygame.draw.circle(sprite, outline, (center, center), radius, pixel_size)
            
            self._player_sprites[direction] = sprite
    
    def _draw_pixel_outline(self, surface: pygame.Surface, color: Tuple[int, int, int], pixel_size: int):
        """Draw a pixel-perfect outline around non-transparent pixels"""
        width, height = surface.get_size()
        
        # Create a mask of the current content
        for y in range(0, height, pixel_size):
            for x in range(0, width, pixel_size):
                # Check if this pixel is transparent
                try:
                    current_alpha = surface.get_at((x, y))[3]
                except:
                    continue
                    
                if current_alpha > 0:
                    # Check neighbors for outline
                    for dx, dy in [(-pixel_size, 0), (pixel_size, 0), (0, -pixel_size), (0, pixel_size)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            try:
                                neighbor_alpha = surface.get_at((nx, ny))[3]
                                if neighbor_alpha == 0:
                                    pygame.draw.rect(surface, (*color, 255), (nx, ny, pixel_size, pixel_size))
                            except:
                                pass
    
    def _create_collectible_sprites(self, cell_size: int):
        """Create pixel art collectible sprites"""
        size = int(cell_size * self.config.collectible_size_ratio)
        pixel_size = max(1, size // 20)  # Smaller pixels for more detail
        
        # Coin sprite
        self._collectible_sprites['coin'] = self._create_pixel_coin(size, pixel_size)
        
        # Gem sprite
        self._collectible_sprites['gem'] = self._create_pixel_gem(size, pixel_size)
        
        # Star sprite
        self._collectible_sprites['star'] = self._create_pixel_star(size, pixel_size)
    
    def _create_pixel_coin(self, size: int, pixel_size: int) -> pygame.Surface:
        """Create a chunky pixel art coin matching Viking style"""
        sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Use larger pixels for chunky look
        p = max(2, size // 8)
        
        color = self.config.coin_color
        highlight = tuple(min(255, c + 50) for c in color)
        shadow = tuple(max(0, c - 50) for c in color)
        dark = tuple(max(0, c - 30) for c in color)
        
        center = size // 2
        
        # Draw chunky circular coin (simplified octagon-like shape)
        # Row by row for pixel-perfect look
        coin_pattern = [
            (2, 4),   # top row: start at 2, width 4
            (1, 6),   # second row
            (0, 8),   # full width rows
            (0, 8),
            (0, 8),
            (0, 8),
            (1, 6),   # narrowing
            (2, 4),   # bottom row
        ]
        
        start_y = center - len(coin_pattern) * p // 2
        
        for row_idx, (indent, width) in enumerate(coin_pattern):
            y = start_y + row_idx * p
            x = center - width * p // 2
            
            for col in range(width):
                px = x + col * p
                
                # Determine color based on position for 3D effect
                if row_idx < 2 or col < 2:
                    c = highlight
                elif row_idx > 5 or col > width - 3:
                    c = shadow
                else:
                    c = color
                
                pygame.draw.rect(sprite, c, (px, y, p, p))
        
        # Inner circle/detail
        inner_size = p * 2
        pygame.draw.rect(sprite, dark, (center - inner_size // 2, center - inner_size // 2, inner_size, inner_size))
        # Shine
        pygame.draw.rect(sprite, highlight, (center - p * 2, center - p * 2, p, p))
        
        return sprite
    
    def _create_pixel_gem(self, size: int, pixel_size: int) -> pygame.Surface:
        """Create a chunky pixel art gem matching Viking style"""
        sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Use larger pixels for chunky look
        p = max(2, size // 8)
        
        color = self.config.gem_color
        highlight = tuple(min(255, c + 60) for c in color)
        shadow = tuple(max(0, c - 50) for c in color)
        
        center = size // 2
        
        # Diamond shape - row by row
        gem_pattern = [
            1,  # top point
            2,
            3,
            4,  # widest
            4,
            3,
            2,
            1,  # bottom point
        ]
        
        start_y = center - len(gem_pattern) * p // 2
        
        for row_idx, width in enumerate(gem_pattern):
            y = start_y + row_idx * p
            x = center - width * p // 2
            
            for col in range(width):
                px = x + col * p
                
                # Top half lighter, bottom half darker
                if row_idx < 4:
                    c = highlight if col < width // 2 else color
                else:
                    c = color if col < width // 2 else shadow
                
                pygame.draw.rect(sprite, c, (px, y, p, p))
        
        # Shine highlight
        pygame.draw.rect(sprite, (255, 255, 255) if highlight[0] > 200 else highlight, 
                        (center - p, start_y + p, p, p))
        
        return sprite
    
    def _create_pixel_star(self, size: int, pixel_size: int) -> pygame.Surface:
        """Create a chunky pixel art star matching Viking style"""
        sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Use larger pixels for chunky look
        p = max(2, size // 8)
        
        color = self.config.star_color
        highlight = tuple(min(255, c + 40) for c in color)
        shadow = tuple(max(0, c - 40) for c in color)
        
        center = size // 2
        
        # 4-pointed star using simple cross pattern
        arm_length = p * 3
        arm_width = p * 2
        
        # Vertical arm
        pygame.draw.rect(sprite, color, 
                        (center - arm_width // 2, center - arm_length, arm_width, arm_length * 2))
        # Horizontal arm
        pygame.draw.rect(sprite, color,
                        (center - arm_length, center - arm_width // 2, arm_length * 2, arm_width))
        
        # Center bright spot
        pygame.draw.rect(sprite, highlight,
                        (center - p, center - p, p * 2, p * 2))
        
        # Top highlight
        pygame.draw.rect(sprite, highlight,
                        (center - arm_width // 2, center - arm_length, arm_width, p))
        # Left highlight
        pygame.draw.rect(sprite, highlight,
                        (center - arm_length, center - arm_width // 2, p, arm_width))
        
        # Bottom shadow
        pygame.draw.rect(sprite, shadow,
                        (center - arm_width // 2, center + arm_length - p, arm_width, p))
        # Right shadow  
        pygame.draw.rect(sprite, shadow,
                        (center + arm_length - p, center - arm_width // 2, p, arm_width))
        
        return sprite
        
    def grid_to_screen(self, grid_x: float, grid_y: float) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates"""
        cell_size = self.config.cell_size
        screen_x = int(self._offset_x + grid_x * cell_size + cell_size // 2)
        screen_y = int(self._offset_y + grid_y * cell_size + cell_size // 2)
        return (screen_x, screen_y)
        
    def is_in_hole(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid position is inside the hole (arrow panel area)"""
        # Check plus-shaped hole
        if hasattr(self, '_hole_plus') and self._hole_plus is not None:
            # Check vertical strip
            if self._hole_plus['vertical'].collidepoint(grid_x, grid_y):
                return True
            # Check horizontal strip
            if self._hole_plus['horizontal'].collidepoint(grid_x, grid_y):
                return True
            return False
        
        # Check rectangular hole
        if self._hole_grid_rect is None:
            return False
        return self._hole_grid_rect.collidepoint(grid_x, grid_y)
        
    def invalidate_maze_cache(self):
        """Mark maze cache as dirty (call when maze changes)"""
        self._maze_dirty = True
        
    def draw_maze(self, screen: pygame.Surface, maze: Maze):
        """
        Draw the maze with hole for arrow panel.
        
        Uses caching for performance since maze rarely changes.
        """
        # Rebuild cache if needed
        if self._maze_dirty:
            self._render_maze_to_cache(maze)
            self._maze_dirty = False
            
        # Blit cached surface
        screen.blit(self._maze_surface, (self._offset_x, self._offset_y))
        
    def _render_maze_to_cache(self, maze: Maze):
        """Render maze to cached surface using pixel art textures, leaving hole transparent"""
        cell_size = self.config.cell_size
        
        # Clear with transparency
        self._maze_surface.fill((0, 0, 0, 0))
        
        # Track if we've drawn the scoreboard (spans 2 cells)
        scoreboard_drawn = False
        
        for y in range(maze.height):
            for x in range(maze.width):
                # Skip cells in the hole area (arrow panel stays unchanged)
                if self.is_in_hole(x, y):
                    continue
                
                # Check if this is a scoreboard cell
                is_scoreboard_cell = hasattr(self, '_scoreboard_cells') and (x, y) in self._scoreboard_cells
                
                pos = (x * cell_size, y * cell_size)
                
                # Determine which texture variant to use for alternating pattern
                use_variant_a = (x + y) % 2 == 0
                
                # Handle scoreboard cells specially
                if is_scoreboard_cell:
                    left_cell = self._scoreboard_cells[0]
                    if (x, y) == left_cell:
                        # Draw floor tiles for both scoreboard cells first
                        texture = self._path_texture_a if use_variant_a else self._path_texture_b
                        self._maze_surface.blit(texture, pos)
                        
                        # Draw floor for right cell too
                        right_pos = ((x + 1) * cell_size, y * cell_size)
                        right_variant = ((x + 1) + y) % 2 == 0
                        right_texture = self._path_texture_a if right_variant else self._path_texture_b
                        self._maze_surface.blit(right_texture, right_pos)
                        
                        # Now draw scoreboard on top of both floor tiles
                        if hasattr(self, '_scoreboard_surface'):
                            self._maze_surface.blit(self._scoreboard_surface, pos)
                            scoreboard_drawn = True
                    # Skip the right cell entirely (already handled above)
                    continue
                    
                cell = maze.get_cell(x, y)
                
                # Draw floor tile first
                texture = self._path_texture_a if use_variant_a else self._path_texture_b
                self._maze_surface.blit(texture, pos)
                
                # Use pixel art textures - skip walls (they show as background)
                if cell == CellType.WALL:
                    # Don't render walls - game field bounded by screen only
                    pass
                elif cell == CellType.START:
                    texture = self._start_texture_a if use_variant_a else self._start_texture_b
                    self._maze_surface.blit(texture, pos)
                elif cell == CellType.GOAL:
                    texture = self._goal_texture_a if use_variant_a else self._goal_texture_b
                    self._maze_surface.blit(texture, pos)
                # PATH cells already drawn above
                    
    def draw_player(self, screen: pygame.Surface, player: Player):
        """Draw the player using pixel art sprite (skip if in hole area)"""
        cell_size = self.config.cell_size
        size = int(cell_size * self.config.player_size_ratio)
        
        # Get interpolated position
        grid_x, grid_y = player.render_position
        
        # Skip if in hole
        if self.is_in_hole(int(grid_x), int(grid_y)):
            return
            
        screen_x, screen_y = self.grid_to_screen(grid_x, grid_y)
        
        # Get sprite based on direction
        from config import Direction
        direction = player.current_direction
        
        if direction == Direction.UP:
            sprite_key = 'up'
        elif direction == Direction.DOWN:
            sprite_key = 'down'
        elif direction == Direction.LEFT:
            sprite_key = 'left'
        elif direction == Direction.RIGHT:
            sprite_key = 'right'
        else:
            sprite_key = 'idle'
        
        sprite = self._player_sprites.get(sprite_key, self._player_sprites.get('idle'))
        
        if sprite:
            # Center the sprite on the position
            sprite_rect = sprite.get_rect(center=(screen_x, screen_y))
            screen.blit(sprite, sprite_rect)
            
    def _draw_direction_indicator(
        self, 
        screen: pygame.Surface,
        cx: int, cy: int, 
        direction, 
        radius: int
    ):
        """Draw a small indicator showing movement direction"""
        from config import Direction
        
        # Direction to angle (in radians)
        angles = {
            Direction.UP: -math.pi / 2,
            Direction.DOWN: math.pi / 2,
            Direction.LEFT: math.pi,
            Direction.RIGHT: 0,
        }
        angle = angles.get(direction, 0)
        
        # Calculate indicator position
        indicator_dist = radius * 0.6
        ix = int(cx + math.cos(angle) * indicator_dist)
        iy = int(cy + math.sin(angle) * indicator_dist)
        
        # Draw small circle
        pygame.draw.circle(
            screen,
            self.config.player_direction_color,
            (ix, iy),
            radius // 4
        )
        
    def draw_collectibles(self, screen: pygame.Surface, manager: CollectibleManager):
        """Draw all collectibles (skip those in hole)"""
        for item in manager.active_collectibles:
            # Skip if in hole
            if self.is_in_hole(item.x, item.y):
                continue
            self._draw_collectible(screen, item)
            
    def _draw_collectible(
        self, 
        screen: pygame.Surface, 
        item: Collectible
    ):
        """Draw a single collectible using pixel art sprites (static, no animation)"""
        # Get position (no animation offset - static items)
        screen_x, screen_y = self.grid_to_screen(item.x, item.y)
        
        # Get the appropriate sprite
        if item.type == CollectibleType.COIN:
            sprite = self._collectible_sprites.get('coin')
        elif item.type == CollectibleType.GEM:
            sprite = self._collectible_sprites.get('gem')
        elif item.type == CollectibleType.STAR:
            sprite = self._collectible_sprites.get('star')
        else:
            sprite = self._collectible_sprites.get('coin')  # Default
        
        if sprite:
            # Center the sprite on the position
            sprite_rect = sprite.get_rect(center=(screen_x, screen_y))
            screen.blit(sprite, sprite_rect)
            
    def _draw_coin(self, screen: pygame.Surface, cx: int, cy: int, size: int):
        """Draw a coin using pixel art sprite"""
        sprite = self._collectible_sprites.get('coin')
        if sprite:
            sprite_rect = sprite.get_rect(center=(cx, cy))
            screen.blit(sprite, sprite_rect)
        
    def _draw_gem(
        self, screen: pygame.Surface, 
        cx: int, cy: int, size: int,
        current_time: float, config
    ):
        """Draw a gem using pixel art sprite"""
        sprite = self._collectible_sprites.get('gem')
        if sprite:
            sprite_rect = sprite.get_rect(center=(cx, cy))
            screen.blit(sprite, sprite_rect)
        
    def _draw_star(
        self, screen: pygame.Surface, 
        cx: int, cy: int, size: int,
        current_time: float, config
    ):
        """Draw a star using pixel art sprite"""
        sprite = self._collectible_sprites.get('star')
        if sprite:
            sprite_rect = sprite.get_rect(center=(cx, cy))
            screen.blit(sprite, sprite_rect)
            
    def draw_ui(
        self, 
        screen: pygame.Surface, 
        score: int,
        collected: int,
        total: int,
        level: int = 1,
        rect: pygame.Rect = None
    ):
        """
        Draw game UI overlay - pixel art scoreboard in top-right corner.
        
        Args:
            screen: Surface to draw on
            score: Current score
            collected: Number of items collected
            total: Total number of items
            level: Current level number
            rect: Optional rect to avoid (e.g., arrow panel)
        """
        cell_size = self.config.cell_size
        
        # Calculate scoreboard position (top-right, 2 cells wide)
        # Get the position of the scoreboard cells
        if hasattr(self, '_scoreboard_cells') and self._scoreboard_cells:
            left_cell = self._scoreboard_cells[0]
            # Convert grid coords to screen coords
            board_x = self._offset_x + left_cell[0] * cell_size
            board_y = self._offset_y + left_cell[1] * cell_size
        else:
            # Fallback position
            board_x = self._offset_x + self._maze_width_px - cell_size * 2
            board_y = self._offset_y
        
        # Scoreboard is drawn as part of the maze cache, so we don't need to draw it here
        # Just draw the text on top
        
        # Use smaller pixel size for scoreboard text
        pixel_size = self._scoreboard_pixel_size if hasattr(self, '_scoreboard_pixel_size') else 2
        
        # Text position relative to board
        if hasattr(self, '_board_text_left'):
            text_x = board_x + self._board_text_left
            text_y = board_y + self._board_text_top
        else:
            # Fallback
            text_x = board_x + 10
            text_y = board_y + 10
        
        # Chalk color (grayscale)
        chalk_color = (180, 180, 180)
        
        # Line spacing based on pixel size (5x7 font + spacing)
        line_height = 7 * pixel_size + pixel_size * 2
        
        # Draw score (top line) with label
        self._draw_pixel_text_sized(screen, f"SCORE:{score}", text_x, text_y, chalk_color, pixel_size)
        
        # Draw items (middle line) with label
        text_y += line_height
        self._draw_pixel_text_sized(screen, f"ITEMS:{collected}/{total}", text_x, text_y, chalk_color, pixel_size)
        
        # Draw level (bottom line) with label
        text_y += line_height
        self._draw_pixel_text_sized(screen, f"LVL:{level}", text_x, text_y, chalk_color, pixel_size)
    
    def _draw_pixel_text_sized(self, surface: pygame.Surface, text: str, x: int, y: int, 
                                color: Tuple[int, int, int] = None, pixel_size: int = 2):
        """Draw text using pixel art font with specified pixel size"""
        if color is None:
            color = (180, 180, 180)
        
        char_width = 5 * pixel_size
        char_height = 7 * pixel_size
        spacing = pixel_size  # Space between characters
        
        cursor_x = x
        for char in text.upper():
            pattern = self._pixel_chars.get(char, self._pixel_chars.get(' '))
            if pattern:
                for row_idx, row in enumerate(pattern):
                    for col_idx, pixel in enumerate(row):
                        if pixel == '#':
                            px = cursor_x + col_idx * pixel_size
                            py = y + row_idx * pixel_size
                            pygame.draw.rect(surface, color, (px, py, pixel_size, pixel_size))
            cursor_x += char_width + spacing
        
    def draw_message(
        self, 
        screen: pygame.Surface, 
        message: str,
        sub_message: str = "",
        color: Tuple[int, int, int] = None
    ):
        """
        Draw a centered message (e.g., "Level Complete!")
        
        Args:
            screen: Surface to draw on
            message: Main message text
            sub_message: Smaller text below
            color: Text color (default: highlight color)
        """
        color = color or self.config.ui_highlight_color
        
        # Get screen center
        screen_rect = screen.get_rect()
        cx, cy = screen_rect.center
        
        # Draw semi-transparent backdrop
        backdrop_width = 400
        backdrop_height = 100
        backdrop = pygame.Surface((backdrop_width, backdrop_height), pygame.SRCALPHA)
        backdrop.fill((0, 0, 0, 180))
        backdrop_rect = backdrop.get_rect(center=(cx, cy))
        screen.blit(backdrop, backdrop_rect)
        
        # Draw main message
        text = self._font_large.render(message, True, color)
        text_rect = text.get_rect(center=(cx, cy - 15))
        screen.blit(text, text_rect)
        
        # Draw sub-message
        if sub_message:
            sub_text = self._font_small.render(sub_message, True, self.config.ui_text_color)
            sub_rect = sub_text.get_rect(center=(cx, cy + 20))
            screen.blit(sub_text, sub_rect)
            
    def get_maze_rect(self) -> pygame.Rect:
        """Get the screen rectangle occupied by the maze"""
        return pygame.Rect(
            self._offset_x,
            self._offset_y,
            self._maze_width_px,
            self._maze_height_px
        )
        
    def get_hole_grid_rect(self) -> Optional[pygame.Rect]:
        """Get the grid rectangle of the hole (for maze generation)"""
        return self._hole_grid_rect


# =============================================================================
# Testing
# =============================================================================

def demo():
    """Visual demo of the renderer"""
    pygame.init()
    
    from src.game.maze import Maze, MazeConfig
    from src.game.player import Player, PlayerConfig
    from src.game.collectible import CollectibleManager, CollectibleConfig
    from config import Direction
    
    # Setup
    screen_width, screen_height = 1024, 768
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Game Renderer Demo")
    clock = pygame.time.Clock()
    
    # Simulate arrow panel in center
    arrow_panel_rect = pygame.Rect(
        screen_width // 2 - 200,
        screen_height // 2 - 150,
        400, 300
    )
    
    # Create game elements
    maze_config = MazeConfig(width=31, height=23, cell_size=32, seed=42)
    maze = Maze(maze_config)
    maze.generate()
    
    player_config = PlayerConfig()
    player = Player(player_config, start_pos=maze.start_pos)
    
    collectible_config = CollectibleConfig()
    collectibles = CollectibleManager(collectible_config)
    collectibles.spawn_random(
        lambda exclude=None: maze.get_random_path_cell(exclude),
        count=15
    )
    
    # Create renderer
    render_config = RenderConfig(cell_size=32)
    renderer = GameRenderer(render_config)
    renderer.setup(screen_width, screen_height, maze, arrow_panel_rect)
    
    running = True
    score = 0
    
    print("Renderer Demo (with hole)")
    print("  Arrow keys: Move player")
    print("  R: Regenerate maze")
    print("  ESC: Quit")
    
    while running:
        delta_ms = clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Regenerate
                    maze.generate()
                    renderer.invalidate_maze_cache()
                    player.set_position(*maze.start_pos)
                    collectibles.clear()
                    collectibles.spawn_random(
                        lambda exclude=None: maze.get_random_path_cell(exclude),
                        count=15
                    )
                    score = 0
                    
                # Movement
                direction = None
                if event.key == pygame.K_UP:
                    direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    direction = Direction.RIGHT
                    
                if direction and player.can_move(direction, maze.is_walkable):
                    player.move(direction)
                    
        # Update
        player.update(delta_ms)
        
        # Check collectible pickup
        px, py = player.grid_position
        collected = collectibles.collect_at(px, py)
        if collected:
            score = collectibles.total_score
            
        # Draw
        screen.fill((20, 20, 20))
        renderer.draw_maze(screen, maze)
        renderer.draw_collectibles(screen, collectibles)
        renderer.draw_player(screen, player)
        
        # Draw placeholder for arrow panel
        pygame.draw.rect(screen, (40, 40, 40), arrow_panel_rect)
        pygame.draw.rect(screen, (80, 80, 80), arrow_panel_rect, 2)
        
        renderer.draw_ui(
            screen, 
            score=score,
            collected=collectibles.collected_count,
            total=collectibles.total_count
        )
        
        # Win message
        if collectibles.all_collected():
            renderer.draw_message(
                screen, 
                "Level Complete!",
                "Press R to play again"
            )
            
        pygame.display.flip()
        
    pygame.quit()
    

if __name__ == "__main__":
    demo()