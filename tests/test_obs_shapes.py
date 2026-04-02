import numpy as np
import pytest
from envs import utils_observation

def test_obs_shape_basic():
    config = {'K': 2} # 6 base + 2*6 = 18
    shape = utils_observation.get_observation_space_shape(config)
    assert shape == (18,)

def test_build_observation_padding():
    # Config
    config = {
        'K': 3, 
        'd_ref': 100.0, 'v_ref': 10.0,
        'p_ref': 100.0, 'v_ref_rel': 20.0
    }
    
    agent_state = {
        'pos': [0, 0, 10], 'vel': [0, 0, 0], 'yaw': 0.0
    }
    goal_pos = [100, 0, 10]
    
    # Only 1 neighbor provided, but K=3
    neighbors = [
        {'pos': [10, 10, 10], 'vel': [0, 0, 0]}
    ]
    
    obs = utils_observation.build_observation(agent_state, neighbors, goal_pos, config)
    
    # Expected size: 6 (own) + 6 (neighbor 1) + 6 (pad 1) + 6 (pad 2) = 24 ??
    # Wait, code: 6 basic + K * neighbors.
    # K=3. So 6 + 3*6 = 24.
    assert obs.shape == (24,)
    
    # Verify padding is zero
    # Last 12 elements should be zero (2 missing neighbors)
    assert np.allclose(obs[-12:], 0.0)

def test_build_observation_values():
    config = {
        'K': 1, 
        'd_ref': 100.0, 'v_ref': 10.0,
        'p_ref': 100.0, 'v_ref_rel': 20.0
    }
    # Drone at origin, Goal at (100, 0, 0)
    # Distance 100. wp_dist_norm = tanh(100/100) = tanh(1) ~ 0.7615
    agent_state = {'pos': [0,0,0], 'vel': [0,0,0], 'yaw': 0.0}
    goal_pos = [100, 0, 0]
    neighbors = []
    
    obs = utils_observation.build_observation(agent_state, neighbors, goal_pos, config)
    
    assert np.isclose(obs[0], np.tanh(1.0), atol=1e-4) # dist
    assert np.isclose(obs[1], 1.0, atol=1e-4)          # cos (bearing X)
    assert np.isclose(obs[2], 0.0, atol=1e-4)          # sin (bearing Y)
