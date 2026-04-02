"""
Smoke test for observation dimension fix
Tests the environment with actual reset/step to ensure 72-dim observations
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.universal_uav_env import UniversalUAVEnv


def test_env_smoke():
    """Smoke test with actual environment"""
    print("Initializing environment...")
    env = UniversalUAVEnv(smoke_test=True)  # Use mock AirSim
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print(f"\nAgent count: {len(obs)}")
    for agent, agent_obs in obs.items():
        print(f"  {agent}: obs shape = {agent_obs.shape}, dims = {len(agent_obs)}")
        assert len(agent_obs) == 103, f"Expected 103 dims, got {len(agent_obs)} for {agent}"
        assert np.all(np.isfinite(agent_obs)), f"Observation contains NaN/inf for {agent}"
    
    print("\nRunning 1000 steps (stress test)...")
    for step in range(1000):
        # Random actions
        actions = {}
        for agent in env.agents:
            if env.use_simplified_actions:
                # [speed, yaw_rate, climb_rate]
                actions[agent] = env.action_space(agent).sample()
            else:
                # [vx, vy, vz, yaw_rate]
                actions[agent] = env.action_space(agent).sample()
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check observation dimensions
        for agent, agent_obs in obs.items():
            if len(agent_obs) != 103:
                print(f"\n❌ Step {step}: {agent} has {len(agent_obs)} dims (expected 103)")
                raise AssertionError(f"Dimension mismatch at step {step}")
        
        # Progress reporting every 100 steps
        if (step + 1) % 100 == 0:
            print(f"  Steps {step+1-99}-{step+1}: ✓ All observations are 103 dims")
        
        # Stop if all agents done
        if not env.agents:
            print("  All agents terminated/truncated")
            break
    
    print("\n✅ Stress test passed! 1000 steps completed.")
    print("   All observations are consistently 103 dimensions.")
    print("   The Cooperative RL dimension change is VERIFIED!")


if __name__ == "__main__":
    test_env_smoke()
