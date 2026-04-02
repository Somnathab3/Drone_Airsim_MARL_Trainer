import os
import sys
import argparse
import yaml
import ray
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Ensure we can import from local envs/ folder if running from root
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from envs.universal_uav_env import UniversalUAVEnv
from training.callbacks import UAveMetricsCallbacks

def env_creator(env_config):
    # env_config contains config_path and smoke_test flag
    config_path = env_config.get("config_path", "config/env.yaml")
    smoke_test = env_config.get("smoke_test", False)
    
    # Create PettingZoo environment
    pz_env = UniversalUAVEnv(config_path=config_path, smoke_test=smoke_test)
    
    # Wrap in ParallelPettingZooEnv for compatibility with Ray's new API stack
    return ParallelPettingZooEnv(pz_env)

register_env("universal_uav_env", env_creator)

def train_rllib(smoke_test=False, debug_iterations=None):
    # Create timestamped run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    checkpoint_dir = os.path.join("models", "checkpoints", run_name)
    episode_log_dir = os.path.join("data", "episodes", run_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(episode_log_dir, exist_ok=True)
    
    print(f"\n=== Training Run: {run_name} ===")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Episodes: {episode_log_dir}\n")
    
    # Load configs
    with open("config/training.yaml", "r") as f:
        train_config = yaml.safe_load(f)
        
    training_params = train_config['training']
    run_config = train_config['runners']
    
    # Update episode log directory in env config
    with open("config/env.yaml", "r") as f:
        env_config_data = yaml.safe_load(f)
    env_config_data['logging']['log_dir'] = os.path.abspath(episode_log_dir)
    
    # Save updated config to temp file for this run
    temp_config_path = os.path.join(checkpoint_dir, "env_config.yaml")
    with open(temp_config_path, "w") as f:
        yaml.dump(env_config_data, f)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Policy Mapping
    # Shared policy for all agents
    def policy_mapping_fn(agent_id, episode=None, **kwargs):
        return "shared_policy"

    # Config Builder
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(
            "universal_uav_env",
            env_config={"config_path": temp_config_path, "smoke_test": smoke_test}
        )
        .framework("torch")
        .env_runners(
            # Use 0 workers to force local worker only (single AirSim instance)
            num_env_runners=0,
            sample_timeout_s=180.0,  # Increase timeout for AirSim
        )
        .resources(
            num_gpus=1 if not smoke_test else 0,  # Use 1 GPU for training
            num_cpus_for_local_worker=2,
        )
        .training(
            train_batch_size=training_params.get('train_batch_size', 4000),
            gamma=training_params.get('gamma', 0.99),
            lr=training_params.get('lr', 0.0003),
            # lambda_=training_params.get('lambda', 0.95), # Might be model config or PPO specific
            # kl_coeff=training_params.get('kl_coeff', 0.2),
            # clip_param=training_params.get('clip_param', 0.2),
            # vf_loss_coeff=training_params.get('vf_loss_coeff', 1.0),
            # entropy_coeff=training_params.get('entropy_coeff', 0.01),
            grad_clip=training_params.get('grad_clip', 10.0),
            # sgd_minibatch_size=training_params.get('sgd_minibatch_size', 128),
            # num_sgd_iter=training_params.get('num_sgd_iter', 10),
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .callbacks(UAveMetricsCallbacks)
    )
    
    # Set PPO specific (or algorithm specific) params directly
    config.sgd_minibatch_size = training_params.get('sgd_minibatch_size', 128)
    config.num_sgd_iter = training_params.get('num_sgd_iter', 10)
    config.lambda_ = training_params.get('lambda', 0.95)
    config.kl_coeff = training_params.get('kl_coeff', 0.2)
    config.clip_param = training_params.get('clip_param', 0.2)
    config.vf_loss_coeff = training_params.get('vf_loss_coeff', 1.0)
    config.entropy_coeff = training_params.get('entropy_coeff', 0.01)

        # .checkpoints(
        #     checkpoint_frequency=10,
        #     local_dir="models/checkpoints"
        # )
    # )
    
    if smoke_test:
        config.train_batch_size = 200
        config.sgd_minibatch_size = 50
    
    # Build Algorithm
    algo = config.build()
    
    print("Starting training Loop...")
    
    # Training loop based on timesteps
    if debug_iterations:
        stop_iters = debug_iterations
        target_timesteps = None
        print(f"DEBUG MODE: Running only {debug_iterations} iterations")
    elif smoke_test:
        stop_iters = 5
        target_timesteps = None
    else:
        # Calculate iterations needed for 1M timesteps
        # With 10 agents, max_steps=1000, train_batch_size=4000
        # Approximate: 1M / 4000 = 250 iterations
        target_timesteps = training_params.get('total_timesteps', 1_000_000)
        stop_iters = 10000  # Safety limit
    
    timesteps_sampled = 0
    
    # Infinite loop that breaks when target timesteps reached
    while True:
        result = algo.train()
        
        # Track timesteps
        if not smoke_test:
            timesteps_sampled = result.get('num_env_steps_sampled_lifetime', timesteps_sampled)
            
            # Robust check: if lifetime steps is 0 (first iter), try falling back to agent steps or accumulating manually
            if timesteps_sampled == 0 and 'env_runners' in result:
                # Try to get it from counters if top-level is missing
                timesteps_sampled = result.get('counters', {}).get('num_env_steps_sampled_lifetime', 0)
        else:
            # Fake increment for smoke test
            timesteps_sampled += 4000
        
        # Debug: print available keys on first iteration in smoke test
        if smoke_test and timesteps_sampled < 5000:
            print("\nAvailable result keys:", sorted(result.keys()))
            print("env_runners:", result.get("env_runners", {}))
            print("sampler_results:", result.get("sampler_results", {}))
            print()

        # Robust fallback chain for reward metrics across RLlib versions
        reward = (
            result.get("episode_reward_mean")
            or result.get("env_runners", {}).get("episode_return_mean")
            or result.get("env_runners", {}).get("episode_reward_mean")
            or result.get("sampler_results", {}).get("episode_reward_mean")
        )
        
        print(f"Iter: {result['training_iteration']} | Steps: {timesteps_sampled:,} | Reward: {reward}")
        
        # Checkpoint every 50 iterations (not in smoke test)
        if result['training_iteration'] % 50 == 0 and not smoke_test and not debug_iterations:
            saved_checkpoint = algo.save(checkpoint_dir=os.path.abspath(checkpoint_dir))
            print(f"[OK] Checkpoint saved at {saved_checkpoint}")
        
        # Stop if we've reached target timesteps
        if target_timesteps and timesteps_sampled >= target_timesteps:
            print(f"\n[OK] Reached target of {target_timesteps:,} timesteps!")
            # Save final checkpoint
            saved_checkpoint = algo.save(checkpoint_dir=os.path.abspath(checkpoint_dir))
            print(f"[OK] Final checkpoint saved at {saved_checkpoint}")
            break
            
        # Safety break for smoke test
        if smoke_test and result['training_iteration'] >= stop_iters:
            break
        
        # Safety break for debug mode
        if debug_iterations and result['training_iteration'] >= debug_iterations:
            print(f"\n[OK] Debug mode: Completed {debug_iterations} iterations")
            saved_checkpoint = algo.save(checkpoint_dir=os.path.abspath(checkpoint_dir))
            print(f"[OK] Checkpoint saved at {saved_checkpoint}")
            break
            
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a short smoke test")
    parser.add_argument("--debug-iterations", type=int, default=None, help="Run only N iterations for debugging")
    args = parser.parse_args()
    
    # Override for debug mode
    if args.debug_iterations:
        train_rllib(smoke_test=False, debug_iterations=args.debug_iterations)
    else:
        train_rllib(smoke_test=args.smoke_test, debug_iterations=None)
