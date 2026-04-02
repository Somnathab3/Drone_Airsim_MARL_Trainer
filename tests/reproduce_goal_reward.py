import numpy as np
import pytest
import sys
import os

# Add parent directory to path to import envs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.universal_uav_env import UniversalUAVEnv

def test_goal_reward_cascading():
    print("Initializing environment...")
    # Initialize env with smoke_test=True to use MockAirSimClient
    env = UniversalUAVEnv(smoke_test=True)
    obs, info = env.reset(seed=42)
    
    agent = env.possible_agents[0]
    goal = env.goals[agent]
    radius = env.config['reward']['goal_radius']
    R_goal = env.config['reward']['R_goal']
    
    print(f"Agent: {agent}")
    print(f"Goal: {goal}")
    print(f"Goal Radius: {radius}")
    print(f"R_goal: {R_goal}")
    
    # Place drone AT goal (z slightly adjusted to match goal exactly or within radius)
    # Goal in env is center, radius is sphere.
    env.airsim_client.states[agent]['pos'] = np.array(goal, dtype=np.float32)
    env.airsim_client.states[agent]['vel'] = np.zeros(3, dtype=np.float32)
    
    # CRITICAL: Update hover target so PID doesn't pull it back to spawn
    env.hover_targets[agent] = np.array(goal, dtype=np.float32)
    
    # Step 1: Should get reward
    print("\n--- Step 1: Placing at goal ---")
    actions = {agent: np.zeros(3) for agent in env.agents} # Hover (simplified action space: speed, yaw_rate, climb_rate)
    
    # We need to ensure the env updates the state from the mock client
    # The step() method calls get_drone_states() which returns the manipulated state
    
    obs, rewards, terms, truncs, infos = env.step(actions)
    
    print(f"Reward 1: {rewards[agent]}")
    print(f"Info 1: {infos[agent]}")
    print(f"Success Set: {env.success_agents}")
    
    # Verify we got the goal reward (approx R_goal, maybe minus small penalties)
    # But definitely > R_goal - 20
    assert rewards[agent] > (R_goal - 20), f"Expected high reward ~{R_goal}, got {rewards[agent]}"
    assert infos[agent].get('outcome') == 'success'
    assert agent in env.success_agents
    
    # Step 2: Still at goal. Should NOT get R_goal again.
    print("\n--- Step 2: Staying at goal ---")
    obs, rewards, terms, truncs, infos = env.step(actions)
    
    print(f"Reward 2: {rewards[agent]}")
    print(f"Outcome 2: {infos[agent].get('outcome', 'running')}")
    
    # Verification Logic
    # If bug is present, Reward 2 will be ~100.
    # If bug is fixed, Reward 2 should be small (~0 or small hover costs).
    
    if rewards[agent] > (R_goal - 20):
        print("FAIL: Reward 2 is still high! Fix NOT working.")
        # This assert is what we expect to FAIL if fix is broken
        pytest.fail(f"Fix failed: Reward 2 is {rewards[agent]}, expected small value")
    else:
        print("PASS: Reward 2 is small. Fix verified.")
        assert rewards[agent] < (R_goal / 2), f"Reward 2 should be small, got {rewards[agent]}"

if __name__ == "__main__":
    try:
        test_goal_reward_cascading()
        print("\nTest passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
