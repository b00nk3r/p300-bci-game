"""
Collectible Items
=================
Items that can be collected by the player for points.

Features:
- Multiple collectible types with different point values
- Simple animation (bobbing, rotation)
- Spawn management
"""

import math
import time
from enum import Enum, auto
from typing import Tuple, List, Optional, Set
from dataclasses import dataclass, field


class CollectibleType(Enum):
    """Types of collectible items"""
    COIN = auto()       # Basic collectible
    GEM = auto()        # Higher value
    STAR = auto()       # Rare, high value
    POWERUP = auto()    # Special effect (future use)


@dataclass
class CollectibleConfig:
    """Configuration for collectibles"""
    # Point values
    coin_points: int = 10
    gem_points: int = 25
    star_points: int = 50
    powerup_points: int = 0  # Powerups give effects, not points
    
    # Appearance
    size_ratio: float = 0.5  # Size relative to cell size
    
    # Animation
    bob_amplitude: float = 0.1    # Bobbing amplitude (fraction of cell)
    bob_speed: float = 2.0        # Bobbing cycles per second
    rotation_speed: float = 90.0  # Degrees per second (for gems/stars)
    
    # Colors (dull grayscale to make arrows stand out)
    coin_color: Tuple[int, int, int] = (75, 75, 75)      # Dull gray
    gem_color: Tuple[int, int, int] = (60, 60, 60)       # Dull gray
    star_color: Tuple[int, int, int] = (80, 80, 80)      # Dull gray
    powerup_color: Tuple[int, int, int] = (70, 70, 70)   # Dull gray


@dataclass
class Collectible:
    """
    A single collectible item.
    
    Attributes:
        x, y: Grid position
        type: Type of collectible
        collected: Whether item has been collected
    """
    x: int
    y: int
    type: CollectibleType = CollectibleType.COIN
    collected: bool = False
    
    # Animation state
    _spawn_time: float = field(default_factory=time.time)
    
    def get_animation_offset(self, current_time: float, config: CollectibleConfig) -> float:
        """
        Get vertical bobbing offset for animation.
        
        Args:
            current_time: Current time in seconds
            config: Collectible configuration
            
        Returns:
            Vertical offset as fraction of cell size
        """
        elapsed = current_time - self._spawn_time
        return math.sin(elapsed * config.bob_speed * 2 * math.pi) * config.bob_amplitude
        
    def get_rotation(self, current_time: float, config: CollectibleConfig) -> float:
        """
        Get rotation angle for spinning items.
        
        Args:
            current_time: Current time in seconds
            config: Collectible configuration
            
        Returns:
            Rotation angle in degrees
        """
        elapsed = current_time - self._spawn_time
        return (elapsed * config.rotation_speed) % 360
        
    @property
    def position(self) -> Tuple[int, int]:
        """Get grid position"""
        return (self.x, self.y)


class CollectibleManager:
    """
    Manages all collectibles in the game.
    
    Features:
    - Spawn collectibles at random positions
    - Track collected items
    - Calculate total score
    
    Usage:
        manager = CollectibleManager(config)
        manager.spawn_random(maze, count=10)
        
        # Check for collection
        collected = manager.collect_at(player_x, player_y)
        if collected:
            points = manager.get_points(collected.type)
            
        # Get all for rendering
        for item in manager.active_collectibles:
            render(item)
    """
    
    def __init__(self, config: CollectibleConfig = None):
        self.config = config or CollectibleConfig()
        
        # All collectibles
        self._collectibles: List[Collectible] = []
        
        # Score tracking
        self._total_score: int = 0
        self._collected_count: int = 0
        
    def clear(self):
        """Remove all collectibles"""
        self._collectibles.clear()
        self._total_score = 0
        self._collected_count = 0
        
    def spawn(self, x: int, y: int, type: CollectibleType = CollectibleType.COIN):
        """
        Spawn a collectible at position.
        
        Args:
            x, y: Grid position
            type: Type of collectible
        """
        collectible = Collectible(x=x, y=y, type=type)
        self._collectibles.append(collectible)
        
    def spawn_random(
        self, 
        get_random_pos_func,
        count: int = 10,
        type_weights: dict = None
    ):
        """
        Spawn collectibles at random positions.
        
        Args:
            get_random_pos_func: Function that returns random (x, y) position
            count: Number of collectibles to spawn
            type_weights: Dict mapping CollectibleType to spawn weight
        """
        if type_weights is None:
            type_weights = {
                CollectibleType.COIN: 0.7,
                CollectibleType.GEM: 0.2,
                CollectibleType.STAR: 0.1,
            }
            
        # Normalize weights
        total_weight = sum(type_weights.values())
        
        # Track used positions
        used_positions: Set[Tuple[int, int]] = set()
        
        for _ in range(count):
            # Get random position
            pos = get_random_pos_func(exclude=used_positions)
            if pos is None:
                break
                
            used_positions.add(pos)
            
            # Choose type based on weights
            import random
            r = random.random() * total_weight
            cumulative = 0
            chosen_type = CollectibleType.COIN
            
            for ctype, weight in type_weights.items():
                cumulative += weight
                if r <= cumulative:
                    chosen_type = ctype
                    break
                    
            # Spawn
            self.spawn(pos[0], pos[1], chosen_type)
            
    def collect_at(self, x: int, y: int) -> Optional[Collectible]:
        """
        Try to collect item at position.
        
        Args:
            x, y: Grid position
            
        Returns:
            Collected item, or None if nothing at position
        """
        for collectible in self._collectibles:
            if (not collectible.collected and 
                collectible.x == x and collectible.y == y):
                
                collectible.collected = True
                self._collected_count += 1
                self._total_score += self.get_points(collectible.type)
                return collectible
                
        return None
        
    def get_points(self, type: CollectibleType) -> int:
        """Get point value for collectible type"""
        points_map = {
            CollectibleType.COIN: self.config.coin_points,
            CollectibleType.GEM: self.config.gem_points,
            CollectibleType.STAR: self.config.star_points,
            CollectibleType.POWERUP: self.config.powerup_points,
        }
        return points_map.get(type, 0)
        
    @property
    def active_collectibles(self) -> List[Collectible]:
        """Get list of uncollected items"""
        return [c for c in self._collectibles if not c.collected]
        
    @property
    def all_collectibles(self) -> List[Collectible]:
        """Get list of all items (including collected)"""
        return self._collectibles.copy()
        
    @property
    def total_score(self) -> int:
        """Get total score from collected items"""
        return self._total_score
        
    @property
    def collected_count(self) -> int:
        """Get number of collected items"""
        return self._collected_count
        
    @property
    def remaining_count(self) -> int:
        """Get number of remaining items"""
        return len(self.active_collectibles)
        
    @property
    def total_count(self) -> int:
        """Get total number of items"""
        return len(self._collectibles)
        
    def all_collected(self) -> bool:
        """Check if all items have been collected"""
        return self.remaining_count == 0 and self.total_count > 0


# =============================================================================
# Testing
# =============================================================================

def demo():
    """Demo collectible system"""
    config = CollectibleConfig()
    manager = CollectibleManager(config)
    
    # Manual spawning
    manager.spawn(5, 5, CollectibleType.COIN)
    manager.spawn(7, 3, CollectibleType.GEM)
    manager.spawn(10, 8, CollectibleType.STAR)
    
    print("Collectible Demo")
    print(f"  Total items: {manager.total_count}")
    print(f"  Active items: {manager.remaining_count}")
    
    # Collect some
    collected = manager.collect_at(5, 5)
    if collected:
        print(f"  Collected {collected.type.name} at (5, 5)")
        print(f"  Points: {manager.get_points(collected.type)}")
        
    print(f"  Score: {manager.total_score}")
    print(f"  Remaining: {manager.remaining_count}")
    
    # Animation demo
    import time
    current_time = time.time()
    for item in manager.active_collectibles:
        offset = item.get_animation_offset(current_time, config)
        rotation = item.get_rotation(current_time, config)
        print(f"  Item at {item.position}: offset={offset:.3f}, rotation={rotation:.1f}°")
        

if __name__ == "__main__":
    demo()
