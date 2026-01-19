"""Tests for configuration module"""

import pytest
from config import Config, TimingConfig, Direction


def test_default_config():
    """Test default configuration values"""
    config = Config()
    
    assert config.timing.flash_duration_ms == 100
    assert config.timing.isi_ms == 125
    assert config.timing.soa_ms == 225
    

def test_timing_soa_calculation():
    """Test SOA is calculated correctly"""
    timing = TimingConfig(flash_duration_ms=100, isi_ms=150)
    assert timing.soa_ms == 250
    

def test_direction_enum():
    """Test Direction enum"""
    directions = Direction.all()
    assert len(directions) == 4
    assert Direction.UP in directions
