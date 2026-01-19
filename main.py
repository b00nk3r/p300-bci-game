#!/usr/bin/env python3
"""
P300 BCI Game - Main Entry Point
================================
Salem State University Capstone Project

Controls:
    SPACE  - Start/Stop BCI selection (arrow flashing)
    S      - Open settings panel
    D      - Toggle debug overlay
    ESC    - Quit
    Arrows - Manual movement (for testing)

Usage:
    python main.py
    python main.py --fullscreen
    python main.py --debug
"""

import sys
import argparse

import pygame

from config import Config, DEFAULT_CONFIG, Direction


class Application:
    """Main application class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.clock = None
        self.screen = None
        
        # Components (to be implemented)
        self.arrow_manager = None
        self.game_manager = None
        self.settings_panel = None
        
        # State
        self.is_selecting = False
        self.show_settings = False
        self.show_debug = config.debug
        
    def initialize(self):
        """Initialize pygame and all components"""
        pygame.init()
        
        # Create display
        flags = pygame.DOUBLEBUF
        if self.config.display.fullscreen:
            flags |= pygame.FULLSCREEN
            
        self.screen = pygame.display.set_mode(
            (self.config.display.width, self.config.display.height),
            flags
        )
        pygame.display.set_caption("P300 BCI Game - Salem State University")
        
        self.clock = pygame.time.Clock()
        
        # Initialize components
        self._init_components()
        
        print("Application initialized successfully!")
        print(f"  Display: {self.config.display.width}x{self.config.display.height}")
        print(f"  Flash duration: {self.config.timing.flash_duration_ms}ms")
        print(f"  ISI: {self.config.timing.isi_ms}ms")
        print(f"  SOA: {self.config.timing.soa_ms}ms ({self.config.timing.flash_rate_hz:.1f}Hz)")
        
    def _init_components(self):
        """Initialize game components"""
        # TODO: Initialize ArrowManager
        # TODO: Initialize GameManager  
        # TODO: Initialize SettingsPanel
        pass
        
    def run(self):
        """Main game loop"""
        self.running = True
        
        while self.running:
            # Handle events
            self._handle_events()
            
            # Update
            self._update()
            
            # Draw
            self._draw()
            
            # Cap framerate
            self.clock.tick(self.config.display.fps)
            
        self._cleanup()
        
    def _handle_events(self):
        """Process input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
                
    def _handle_keydown(self, key: int):
        """Handle keyboard input"""
        if key == pygame.K_ESCAPE:
            self.running = False
            
        elif key == pygame.K_SPACE:
            self._toggle_selection()
            
        elif key == pygame.K_s:
            self.show_settings = not self.show_settings
            
        elif key == pygame.K_d:
            self.show_debug = not self.show_debug
            
        # Manual movement (for testing)
        elif key == pygame.K_UP:
            self._manual_move(Direction.UP)
        elif key == pygame.K_DOWN:
            self._manual_move(Direction.DOWN)
        elif key == pygame.K_LEFT:
            self._manual_move(Direction.LEFT)
        elif key == pygame.K_RIGHT:
            self._manual_move(Direction.RIGHT)
            
    def _toggle_selection(self):
        """Start or stop BCI selection"""
        self.is_selecting = not self.is_selecting
        if self.is_selecting:
            print("BCI Selection started - arrows flashing")
            # TODO: Start arrow flashing
        else:
            print("BCI Selection stopped")
            # TODO: Stop arrow flashing
            
    def _manual_move(self, direction: Direction):
        """Handle manual movement (for testing without EEG)"""
        print(f"Manual move: {direction.value}")
        # TODO: Move player in game
        
    def _update(self):
        """Update game state"""
        # TODO: Update arrow manager
        # TODO: Update game manager
        pass
        
    def _draw(self):
        """Render frame"""
        # Clear screen
        self.screen.fill(self.config.display.background_color)
        
        # TODO: Draw game (maze)
        # TODO: Draw arrow panel
        # TODO: Draw settings if visible
        
        # Draw debug info
        if self.show_debug:
            self._draw_debug()
            
        # Draw placeholder text
        self._draw_placeholder()
            
        # Flip display
        pygame.display.flip()
        
    def _draw_placeholder(self):
        """Draw placeholder text until components are implemented"""
        font = pygame.font.Font(None, 48)
        text = font.render("P300 BCI Game - Press SPACE to start", True, (100, 100, 100))
        rect = text.get_rect(center=(self.config.display.width // 2, 
                                      self.config.display.height // 2))
        self.screen.blit(text, rect)
        
        # Status
        font_small = pygame.font.Font(None, 32)
        status = f"Selection: {'ACTIVE' if self.is_selecting else 'IDLE'}"
        status_text = font_small.render(status, True, (80, 80, 80))
        self.screen.blit(status_text, (20, self.config.display.height - 40))
        
    def _draw_debug(self):
        """Draw debug information"""
        font = pygame.font.Font(None, 24)
        fps = self.clock.get_fps()
        
        debug_lines = [
            f"FPS: {fps:.1f}",
            f"Selection: {self.is_selecting}",
            f"Flash rate: {self.config.timing.flash_rate_hz:.1f}Hz",
        ]
        
        y = 10
        for line in debug_lines:
            text = font.render(line, True, (100, 100, 100))
            self.screen.blit(text, (10, y))
            y += 20
            
    def _cleanup(self):
        """Clean up resources"""
        pygame.quit()
        print("Application closed.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="P300 BCI Game")
    parser.add_argument("--fullscreen", "-f", action="store_true",
                        help="Run in fullscreen mode")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--width", type=int, default=1920,
                        help="Window width")
    parser.add_argument("--height", type=int, default=1080,
                        help="Window height")
    return parser.parse_args()


def main():
    """Entry point"""
    args = parse_args()
    
    # Create configuration
    config = Config()
    config.display.width = args.width
    config.display.height = args.height
    config.display.fullscreen = args.fullscreen
    config.debug = args.debug
    
    # Create and run application
    app = Application(config)
    app.initialize()
    app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
