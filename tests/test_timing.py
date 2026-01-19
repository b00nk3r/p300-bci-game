"""Tests for timing precision"""

import time
import pytest


def test_perf_counter_precision():
    """Verify time.perf_counter has sufficient precision"""
    samples = []
    for _ in range(100):
        start = time.perf_counter()
        # Minimal operation
        end = time.perf_counter()
        samples.append((end - start) * 1_000_000)  # microseconds
        
    # Should have microsecond precision
    avg = sum(samples) / len(samples)
    assert avg < 100  # Less than 100 microseconds overhead
    

def test_busy_wait_accuracy():
    """Test busy-wait timing accuracy"""
    target_ms = 100
    errors = []
    
    for _ in range(10):
        start = time.perf_counter()
        target = start + target_ms / 1000
        while time.perf_counter() < target:
            pass
        elapsed_ms = (time.perf_counter() - start) * 1000
        errors.append(abs(elapsed_ms - target_ms))
        
    avg_error = sum(errors) / len(errors)
    assert avg_error < 1.0  # Less than 1ms average error
