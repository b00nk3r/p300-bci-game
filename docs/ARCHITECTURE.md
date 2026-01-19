# P300 BCI Game Architecture

## Overview

This application implements a P300 Brain-Computer Interface game for 
controlling a maze using EEG signals.

## Module Structure

```
src/
├── stimulus/           # Stimulus presentation
│   ├── arrow_renderer  # Drawing arrows
│   ├── timing_controller # Precise timing
│   └── arrow_manager   # Coordination
├── game/              # Game logic
│   ├── maze           # Maze generation
│   ├── player         # Player state
│   └── game_manager   # Game coordination
├── ui/                # User interface
│   └── settings_panel # Runtime configuration
└── communication/     # External communication
    ├── trigger_manager # EEG triggers
    └── matlab_bridge   # MATLAB interface
```

## Data Flow

```
EEG Recording (g.tec)
        ↓
   [Triggers] ←────── Python Game ──────→ [Display]
        ↓                                     ↑
MATLAB Processing                        User Views
        ↓                                     │
  P300 Classifier                             │
        ↓                                     │
Selection Result ─────────────────────────────┘
```

## Key Design Decisions

1. **Pygame over PsychoPy**: Simpler learning curve, adequate timing
2. **File-based triggers**: Easy MATLAB integration, can upgrade to LSL
3. **Busy-wait timing**: Sub-millisecond precision
4. **Grayscale game**: Minimizes interference with arrow flashes
