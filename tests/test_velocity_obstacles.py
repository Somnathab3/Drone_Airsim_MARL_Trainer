"""
Unit tests for velocity obstacle fixed-length arrays
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.velocity_obstacles import compute_velocity_obstacle_for_neighbors


def test_vo_zero_neighbors():
    """Test with 0 neighbors - should return fixed-length K arrays"""
    K = 5
    pos_self = np.array([0, 0, 0])
    vel_self = np.array([1, 0, 0])
    neighbors = []  # No neighbors
    
    flags, ttcs, cpas = compute_velocity_obstacle_for_neighbors(
        pos_self, vel_self, neighbors, radius=2.0, K=K
    )
    
    assert len(flags) == K, f"Expected {K} flags, got {len(flags)}"
    assert len(ttcs) == K, f"Expected {K} ttcs, got {len(ttcs)}"
    assert len(cpas) == K, f"Expected {K} cpas, got {len(cpas)}"
    
    # All should be safe/no collision
    assert np.all(flags == 0.0), "All flags should be 0.0 (no collision)"
    assert np.all(ttcs == np.inf), "All TTCs should be inf (safe)"
    assert np.all(cpas == np.inf), "All CPAs should be inf (safe)"
    
    print("✓ Zero neighbors test passed")


def test_vo_few_neighbors():
    """Test with fewer than K neighbors"""
    K = 5
    pos_self = np.array([0, 0, 0])
    vel_self = np.array([1, 0, 0])
    neighbors = [
        {'pos': [10, 0, 0], 'vel': [0, 0, 0], 'dist': 10.0, 'is_obstacle': False},
        {'pos': [0, 10, 0], 'vel': [0, 0, 0], 'dist': 10.0, 'is_obstacle': False}
    ]
    
    flags, ttcs, cpas = compute_velocity_obstacle_for_neighbors(
        pos_self, vel_self, neighbors, radius=2.0, K=K
    )
    
    assert len(flags) == K, f"Expected {K} flags, got {len(flags)}"
    assert len(ttcs) == K, f"Expected {K} ttcs, got {len(ttcs)}"
    assert len(cpas) == K, f"Expected {K} cpas, got {len(cpas)}"
    
    # First 2 should have values, rest should be padded
    assert ttcs[2] == np.inf, "Padded TTCs should be inf"
    assert cpas[2] == np.inf, "Padded CPAs should be inf"
    
    print("✓ Few neighbors test passed")


def test_vo_exact_k_neighbors():
    """Test with exactly K neighbors"""
    K = 5
    pos_self = np.array([0, 0, 0])
    vel_self = np.array([1, 0, 0])
    neighbors = [
        {'pos': [i*5, 0, 0], 'vel': [0, 0, 0], 'dist': i*5, 'is_obstacle': False}
        for i in range(1, K+1)
    ]
    
    flags, ttcs, cpas = compute_velocity_obstacle_for_neighbors(
        pos_self, vel_self, neighbors, radius=2.0, K=K
    )
    
    assert len(flags) == K, f"Expected {K} flags, got {len(flags)}"
    assert len(ttcs) == K, f"Expected {K} ttcs, got {len(ttcs)}"
    assert len(cpas) == K, f"Expected {K} cpas, got {len(cpas)}"
    
    print("✓ Exact K neighbors test passed")


def test_vo_more_than_k_neighbors():
    """Test with more than K neighbors - should truncate to K"""
    K = 5
    pos_self = np.array([0, 0, 0])
    vel_self = np.array([1, 0, 0])
    neighbors = [
        {'pos': [i*5, 0, 0], 'vel': [0, 0, 0], 'dist': i*5, 'is_obstacle': False}
        for i in range(1, K+3)  # K+2 neighbors
    ]
    
    flags, ttcs, cpas = compute_velocity_obstacle_for_neighbors(
        pos_self, vel_self, neighbors, radius=2.0, K=K
    )
    
    assert len(flags) == K, f"Expected {K} flags, got {len(flags)}"
    assert len(ttcs) == K, f"Expected {K} ttcs, got {len(ttcs)}"
    assert len(cpas) == K, f"Expected {K} cpas, got {len(cpas)}"
    
    print("✓ More than K neighbors test passed")


def test_vo_output_types():
    """Test that output types are correct (float32)"""
    K = 5
    pos_self = np.array([0, 0, 0])
    vel_self = np.array([1, 0, 0])
    neighbors = []
    
    flags, ttcs, cpas = compute_velocity_obstacle_for_neighbors(
        pos_self, vel_self, neighbors, radius=2.0, K=K
    )
    
    assert flags.dtype == np.float32, f"Flags should be float32, got {flags.dtype}"
    assert ttcs.dtype == np.float64 or ttcs.dtype == np.float32, f"TTCs should be float, got {ttcs.dtype}"
    assert cpas.dtype == np.float64 or cpas.dtype == np.float32, f"CPAs should be float, got {cpas.dtype}"
    
    print("✓ Output type test passed")


if __name__ == "__main__":
    print("Running velocity obstacle tests...")
    test_vo_zero_neighbors()
    test_vo_few_neighbors()
    test_vo_exact_k_neighbors()
    test_vo_more_than_k_neighbors()
    test_vo_output_types()
    print("\n✅ All velocity obstacle tests passed!")
