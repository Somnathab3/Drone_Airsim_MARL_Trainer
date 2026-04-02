"""
Integration test for observation dimension consistency
"""
import numpy as np
import sys
import os
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.utils_observation import build_observation, get_observation_space_shape


def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'env.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_test_agent_state(include_vo=True, vo_length=None):
    """Create a test agent state"""
    K = 5
    state = {
        'pos': [0, 0, 10],
        'vel': [1, 0, 0],
        'yaw': 0.0,
        'centroid_pos': [5, 5, 10],
        'centroid_vel': [0.5, 0, 0],
        'ttc_min': 10.0,
        'progress_rate': 0.1,
        'safety_rate': 0.0,
        'waypoint_direction': [1.0, 0.0, 0.0],
        'waypoint_distance': 20.0,
        'obstacle_rays': np.full(8, 50.0)
    }
    
    if include_vo:
        if vo_length is None:
            vo_length = K
        state['vo_collision_flags'] = np.zeros(vo_length, dtype=np.float32)
        state['vo_ttcs'] = np.full(vo_length, 10.0, dtype=np.float32)
    
    return state


def test_observation_with_zero_neighbors():
    """Test observation with 0 neighbors (62 vs 72 bug scenario)"""
    config = load_config()
    K = config['observation']['K']
    
    # Create state with EMPTY VO arrays (simulates 0 neighbors before fix)
    agent_state = create_test_agent_state(include_vo=True, vo_length=0)
    neighbors = []
    goal_pos = [20, 0, 10]
    
    obs = build_observation(agent_state, neighbors, goal_pos, config['observation'])
    expected_size = get_observation_space_shape(config['observation'])[0]
    
    assert len(obs) == expected_size, \
        f"Expected {expected_size} dims, got {len(obs)} (62 vs 72 bug!)"
    assert np.all(np.isfinite(obs)), "Observation contains NaN or inf"
    
    print(f"✓ Zero neighbors test passed (dims: {len(obs)})")


def test_observation_with_few_neighbors():
    """Test observation with fewer than K neighbors"""
    config = load_config()
    K = config['observation']['K']
    
    # Simulate 2 neighbors (fewer than K=5)
    agent_state = create_test_agent_state(include_vo=True, vo_length=2)
    neighbors = [
        {'pos': [10, 0, 10], 'vel': [0, 0, 0], 'dist': 10.0},
        {'pos': [0, 10, 10], 'vel': [0, 0, 0], 'dist': 10.0}
    ]
    goal_pos = [20, 0, 10]
    
    obs = build_observation(agent_state, neighbors, goal_pos, config['observation'])
    expected_size = get_observation_space_shape(config['observation'])[0]
    
    assert len(obs) == expected_size, \
        f"Expected {expected_size} dims, got {len(obs)}"
    assert np.all(np.isfinite(obs)), "Observation contains NaN or inf"
    
    print(f"✓ Few neighbors test passed (dims: {len(obs)})")


def test_observation_with_k_neighbors():
    """Test observation with exactly K neighbors"""
    config = load_config()
    K = config['observation']['K']
    
    agent_state = create_test_agent_state(include_vo=True, vo_length=K)
    neighbors = [
        {'pos': [i*5, 0, 10], 'vel': [0, 0, 0], 'dist': i*5}
        for i in range(1, K+1)
    ]
    goal_pos = [20, 0, 10]
    
    obs = build_observation(agent_state, neighbors, goal_pos, config['observation'])
    expected_size = get_observation_space_shape(config['observation'])[0]
    
    assert len(obs) == expected_size, \
        f"Expected {expected_size} dims, got {len(obs)}"
    assert np.all(np.isfinite(obs)), "Observation contains NaN or inf"
    
    print(f"✓ K neighbors test passed (dims: {len(obs)})")


def test_observation_with_more_neighbors():
    """Test observation with more than K neighbors"""
    config = load_config()
    K = config['observation']['K']
    
    # Create state with K neighbors (will be padded/truncated internally)
    agent_state = create_test_agent_state(include_vo=True, vo_length=K+2)
    neighbors = [
        {'pos': [i*5, 0, 10], 'vel': [0, 0, 0], 'dist': i*5}
        for i in range(1, K+3)
    ]
    goal_pos = [20, 0, 10]
    
    obs = build_observation(agent_state, neighbors, goal_pos, config['observation'])
    expected_size = get_observation_space_shape(config['observation'])[0]
    
    assert len(obs) == expected_size, \
        f"Expected {expected_size} dims, got {len(obs)}"
    assert np.all(np.isfinite(obs)), "Observation contains NaN or inf"
    
    print(f"✓ More than K neighbors test passed (dims: {len(obs)})")


def test_observation_dimension_formula():
    """Verify observation dimension formula"""
    config = load_config()
    K = config['observation']['K']
    
    # Formula: 4 (goal) + 4 (waypoint) + 3 (vel) + 3 (derivatives) + 6 (centroid) + 4 (prev_action) + 8 (rays) + K (VO flags) + K (TTCs) + K*6 (neighbors)
    expected = 4 + 4 + 3 + 3 + 6 + 4 + 8 + K + K + (K * 6)
    actual = get_observation_space_shape(config['observation'])[0]
    
    assert expected == actual, f"Dimension formula mismatch: expected {expected}, got {actual}"
    print(f"✓ Dimension formula verified: 32 + 8*K = 32 + 8*{K} = {actual}")


if __name__ == "__main__":
    print("Running observation integration tests...")
    print(f"Testing with K={load_config()['observation']['K']}")
    print()
    
    test_observation_dimension_formula()
    test_observation_with_zero_neighbors()
    test_observation_with_few_neighbors()
    test_observation_with_k_neighbors()
    test_observation_with_more_neighbors()
    
    print("\n✅ All observation integration tests passed!")
