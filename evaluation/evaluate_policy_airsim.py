import argparse
import ray
from ray.rllib.algorithms.ppo import PPO
from envs.universal_uav_env import UniversalUAVEnv
from ray.tune.registry import register_env
import time
import yaml
import os

def env_creator(config):
    return UniversalUAVEnv(config_path="config/env.yaml")

register_env("universal_uav_env", env_creator)

def evaluate(checkpoint_path, num_episodes=5):
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Load env config to get observation/action spaces logic if needed,
    # but PPO load should handle it if env is registered.
    
    # Restore agent
    # We need to reconstruct the config to match training or just load checkpoint.
    # algo = PPO.from_checkpoint(checkpoint_path) # RLlib 2.x
    
    # For older RLlib/Ray versions, might need Algorithm.from_checkpoint
    from ray.rllib.algorithms.algorithm import Algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Environment
    env = UniversalUAVEnv(config_path="config/env.yaml")
    
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Starting evaluation for {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = {agent: False for agent in env.agents}
        total_rewards = {agent: 0.0 for agent in env.agents}
        steps = 0
        
        while not all(done.values()):
            # Compute actions
            actions = {}
            for agent in env.agents:
                if done[agent]: continue
                
                # Policy mapping: all use "shared_policy"
                # compute_single_action(observation, policy_id=...)
                action = algo.compute_single_action(
                    obs[agent], 
                    policy_id="shared_policy",
                    explore=False # Deterministic for evaluation
                )
                actions[agent] = action
            
            # Step
            next_obs, rewards, term, trunc, infos = env.step(actions)
            
            for agent in env.agents:
                total_rewards[agent] += rewards.get(agent, 0.0)
                if term.get(agent, False) or trunc.get(agent, False):
                    done[agent] = True
            
            obs = next_obs
            steps += 1
            # Optional: slow down for visualization
            # time.sleep(0.05)
            
        print(f"Episode {ep} finished. Steps: {steps}. Rewards: {total_rewards}")
        
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.episodes)
