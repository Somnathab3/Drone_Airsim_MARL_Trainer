import numpy as np
import pytest
from envs import utils_reward

def test_reward_progress():
    # Agent moved closer to goal
    # d_prev = 100, dist = 90. Progress = 10.
    # r = alpha * 10
    config = {'alpha': 1.0, 'eta': 0.0, 'R_collision': 0.0, 'R_goal': 0.0}
    
    prev_state = {'dist_to_goal': 100.0}
    agent_state = {'pos': [10, 0, 0], 'collision': False}
    goal_pos = [20, 0, 0] # dist = 10
    
    info = {}
    reward, term, trunc = utils_reward.compute_reward(
        agent_state, goal_pos, prev_state, config, info
    )
    
    # Progress = 1.0 * (100 - 10) = 90
    assert np.isclose(reward, 90.0)
    assert not term

def test_reward_collision():
    config = {'R_collision': 50.0, 'alpha': 0.0, 'eta': 0.0}
    agent_state = {'pos': [0,0,0], 'collision': True}
    goal_pos = [100,0,0]
    prev_state = {'dist_to_goal': 100.0}
    
    info = {}
    reward, term, trunc = utils_reward.compute_reward(
        agent_state, goal_pos, prev_state, config, info
    )
    
    # Reward = -50
    assert reward == -50.0
    assert term
    assert info['outcome'] == 'collision'

def test_reward_goal():
    config = {'R_goal': 100.0, 'goal_radius': 5.0, 'alpha': 0.0, 'eta': 0.0}
    agent_state = {'pos': [98,0,0], 'collision': False} # dist 2
    goal_pos = [100,0,0]
    prev_state = {'dist_to_goal': 10.0}
    
    info = {}
    reward, term, trunc = utils_reward.compute_reward(
        agent_state, goal_pos, prev_state, config, info
    )
    
    assert reward == 100.0
    assert term
    assert info['outcome'] == 'success'
