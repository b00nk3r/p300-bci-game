"""
Mode Selector
=============
Provides mode selection UI for choosing between:
- Data Collection Mode: For BCI training data collection
- Game Mode: For playing the maze game with BCI control
"""

import pygame
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable


class AppMode(Enum):
    """Application operating modes"""
    DATA_COLLECTION = auto()  # For collecting BCI training data
    GAME = auto()             # For playing the game


@dataclass
class ModeConfig:
    """Configuration for each mode"""
    name: str
    description: str
    color: tuple  # Accent color for mode indicator
    

MODE_CONFIGS = {
    AppMode.DATA_COLLECTION: ModeConfig(
        name="Data Collection",
        description="Collect EEG training data for BCI calibration",
        color=(180, 180, 180),  # Light gray
    ),
    AppMode.GAME: ModeConfig(
        name="Game Mode", 
        description="Play the maze game using BCI control",
        color=(140, 140, 140),  # Medium gray
    ),
}


class ModeSelector:
    """
    Mode selection screen shown at startup.
    
    Allows user to choose between Data Collection and Game modes.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.selected_mode: Optional[AppMode] = None
        self.hovered_mode: Optional[AppMode] = None
        
        # Fonts
        self.title_font = pygame.font.Font(None, 72)
        self.subtitle_font = pygame.font.Font(None, 36)
        self.button_font = pygame.font.Font(None, 48)
        self.desc_font = pygame.font.Font(None, 28)
        self.hint_font = pygame.font.Font(None, 24)
        
        # Colors (grayscale)
        self.bg_color = (20, 20, 20)
        self.title_color = (220, 220, 220)
        self.subtitle_color = (140, 140, 140)
        self.button_color = (50, 50, 50)
        self.button_hover_color = (80, 80, 80)
        self.button_text_color = (200, 200, 200)
        self.desc_color = (120, 120, 120)
        self.hint_color = (80, 80, 80)
        
        # Button dimensions
        self.button_width = 400
        self.button_height = 120
        self.button_spacing = 60
        
        # Calculate button positions
        total_height = 2 * self.button_height + self.button_spacing
        start_y = (screen_height - total_height) // 2 + 50
        
        self.buttons = {
            AppMode.DATA_COLLECTION: pygame.Rect(
                (screen_width - self.button_width) // 2,
                start_y,
                self.button_width,
                self.button_height
            ),
            AppMode.GAME: pygame.Rect(
                (screen_width - self.button_width) // 2,
                start_y + self.button_height + self.button_spacing,
                self.button_width,
                self.button_height
            ),
        }
        
    def handle_event(self, event: pygame.event.Event) -> Optional[AppMode]:
        """
        Handle pygame event.
        
        Returns:
            Selected AppMode if a mode was chosen, None otherwise
        """
        if event.type == pygame.MOUSEMOTION:
            self.hovered_mode = None
            for mode, rect in self.buttons.items():
                if rect.collidepoint(event.pos):
                    self.hovered_mode = mode
                    break
                    
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for mode, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        self.selected_mode = mode
                        return mode
                        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.selected_mode = AppMode.DATA_COLLECTION
                return AppMode.DATA_COLLECTION
            elif event.key == pygame.K_2:
                self.selected_mode = AppMode.GAME
                return AppMode.GAME
                
        return None
        
    def draw(self, screen: pygame.Surface):
        """Draw the mode selection screen"""
        screen.fill(self.bg_color)
        
        # Title
        title = self.title_font.render("P300 BCI Game", True, self.title_color)
        title_rect = title.get_rect(centerx=self.screen_width // 2, top=80)
        screen.blit(title, title_rect)
        
        # Subtitle
        subtitle = self.subtitle_font.render("Salem State University", True, self.subtitle_color)
        subtitle_rect = subtitle.get_rect(centerx=self.screen_width // 2, top=title_rect.bottom + 10)
        screen.blit(subtitle, subtitle_rect)
        
        # Mode selection prompt
        prompt = self.subtitle_font.render("Select Mode", True, self.title_color)
        prompt_rect = prompt.get_rect(centerx=self.screen_width // 2, top=subtitle_rect.bottom + 60)
        screen.blit(prompt, prompt_rect)
        
        # Draw buttons
        for mode, rect in self.buttons.items():
            config = MODE_CONFIGS[mode]
            is_hovered = mode == self.hovered_mode
            
            # Button background
            bg_color = self.button_hover_color if is_hovered else self.button_color
            pygame.draw.rect(screen, bg_color, rect, border_radius=10)
            
            # Button border
            border_color = (150, 150, 150) if is_hovered else (100, 100, 100)
            pygame.draw.rect(screen, border_color, rect, 2, border_radius=10)
            
            # Mode indicator bar on left
            indicator_rect = pygame.Rect(rect.left + 10, rect.top + 10, 6, rect.height - 20)
            pygame.draw.rect(screen, config.color, indicator_rect, border_radius=3)
            
            # Button text
            text = self.button_font.render(config.name, True, self.button_text_color)
            text_rect = text.get_rect(centerx=rect.centerx, centery=rect.centery - 15)
            screen.blit(text, text_rect)
            
            # Description
            desc = self.desc_font.render(config.description, True, self.desc_color)
            desc_rect = desc.get_rect(centerx=rect.centerx, centery=rect.centery + 25)
            screen.blit(desc, desc_rect)
            
        # Keyboard hints
        hint1 = self.hint_font.render("Press 1 for Data Collection", True, self.hint_color)
        hint1_rect = hint1.get_rect(centerx=self.screen_width // 2 - 120, bottom=self.screen_height - 40)
        screen.blit(hint1, hint1_rect)
        
        hint2 = self.hint_font.render("Press 2 for Game Mode", True, self.hint_color)
        hint2_rect = hint2.get_rect(centerx=self.screen_width // 2 + 120, bottom=self.screen_height - 40)
        screen.blit(hint2, hint2_rect)


class ModeIndicator:
    """
    Persistent mode indicator shown during gameplay.
    
    Displays current mode in corner of screen.
    """
    
    def __init__(self, mode: AppMode, screen_width: int, screen_height: int):
        self.mode = mode
        self.config = MODE_CONFIGS[mode]
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        
        # Position (top-left corner)
        self.padding = 15
        
    def draw(self, screen: pygame.Surface):
        """Draw the mode indicator"""
        # Create text
        text = self.font.render(f"Mode: {self.config.name}", True, self.config.color)
        
        # Background
        bg_rect = pygame.Rect(
            self.padding - 5,
            self.padding - 3,
            text.get_width() + 20,
            text.get_height() + 10
        )
        
        # Draw semi-transparent background
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 150))
        screen.blit(bg_surface, bg_rect.topleft)
        
        # Draw indicator bar
        indicator_rect = pygame.Rect(bg_rect.left + 3, bg_rect.top + 3, 3, bg_rect.height - 6)
        pygame.draw.rect(screen, self.config.color, indicator_rect)
        
        # Draw text
        screen.blit(text, (self.padding + 10, self.padding))


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    pygame.init()
    
    screen_width = 1280
    screen_height = 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mode Selector Test")
    
    selector = ModeSelector(screen_width, screen_height)
    clock = pygame.time.Clock()
    
    selected_mode = None
    indicator = None
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if indicator:
                        # Go back to selector
                        indicator = None
                        selected_mode = None
                    else:
                        running = False
                        
            if not selected_mode:
                result = selector.handle_event(event)
                if result:
                    selected_mode = result
                    indicator = ModeIndicator(result, screen_width, screen_height)
                    print(f"Selected mode: {result.name}")
        
        # Draw
        if selected_mode:
            # Show mode indicator on blank screen
            screen.fill((30, 30, 30))
            indicator.draw(screen)
            
            # Show hint
            font = pygame.font.Font(None, 36)
            text = font.render(f"Running in {MODE_CONFIGS[selected_mode].name}", True, (150, 150, 150))
            rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(text, rect)
            
            hint = pygame.font.Font(None, 24).render("Press ESC to go back", True, (80, 80, 80))
            hint_rect = hint.get_rect(center=(screen_width // 2, screen_height // 2 + 40))
            screen.blit(hint, hint_rect)
        else:
            selector.draw(screen)
            
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()