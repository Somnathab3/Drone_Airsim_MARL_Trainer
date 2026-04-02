"""
Verification script to ensure all fixes are properly applied.
"""
import yaml
import json
import numpy as np
from pathlib import Path

def check_config_files():
    """Verify configuration files have correct values."""
    print("="*80)
    print("VERIFYING CONFIGURATION FILES")
    print("="*80)
    
    # Check env.yaml
    with open("config/env.yaml") as f:
        env_config = yaml.safe_load(f)
    
    reward = env_config['reward']
    
    print("\n✅ Reward Coefficients (config/env.yaml):")
    checks = [
        ("alpha (Progress)", reward['alpha'], 20.0, "Increased from 10.0"),
        ("zeta (Heading)", reward['zeta'], 1.5, "NEW - Critical for goal-seeking"),
        ("gamma (Action cost)", reward['gamma'], 0.005, "Reduced from 0.01"),
        ("lambda (Smoothness)", reward['lambda'], 0.1, "Reduced from 0.05"),
        ("beta (Near-miss)", reward['beta'], 0.5, "Reduced from 10.0"),
        ("R_collision", reward['R_collision'], 200.0, "Reduced from 500"),
    ]
    
    all_correct = True
    for name, actual, expected, note in checks:
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_correct = False
        print(f"  {status} {name}: {actual} (expected {expected}) - {note}")
    
    # Check airsim_settings.json
    with open("config/airsim_settings.json") as f:
        airsim_config = json.load(f)
    
    print("\n✅ AirSim Settings (config/airsim_settings.json):")
    clock_speed = airsim_config.get('ClockSpeed', 1)
    view_mode = airsim_config.get('ViewMode', 'SpringArmChase')
    physics_engine = airsim_config.get('PhysicsEngineName', 'Default')
    
    clock_status = "✅" if clock_speed >= 1 else "⚠️"
    view_status = "✅" if view_mode == "" else "⚠️"
    physics_status = "✅" if physics_engine == "FastPhysicsEngine" else "⚠️"
    
    if clock_speed < 1:
        all_correct = False
    
    print(f"  {clock_status} ClockSpeed: {clock_speed} (config value)")
    print(f"  {view_status} ViewMode: '{view_mode}' (empty = headless)")
    print(f"  {physics_status} PhysicsEngine: '{physics_engine}' (FastPhysicsEngine recommended)")
    
    return all_correct

def check_reward_implementation():
    """Verify heading reward is implemented in code."""
    print("\n" + "="*80)
    print("VERIFYING REWARD IMPLEMENTATION")
    print("="*80)
    
    # Read utils_reward.py
    with open("envs/utils_reward.py") as f:
        code = f.read()
    
    checks = [
        ("Heading reward variable (zeta)", "zeta = config.get('zeta'" in code),
        ("Heading alignment calculation", "goal_bearing_enu = np.arctan2" in code),
        ("Heading error wrapping", "heading_error = np.arctan2(np.sin(goal_bearing_enu - current_heading)" in code),
        ("Heading reward formula", "r_heading = zeta * (np.pi - abs(heading_error)) / np.pi" in code),
        ("Heading reward applied", "reward += r_heading" in code),
        ("Heading info logged", "info['r_heading']" in code),
    ]
    
    all_correct = True
    for name, present in checks:
        status = "✅" if present else "❌"
        if not present:
            all_correct = False
        print(f"  {status} {name}")
    
    return all_correct

def estimate_training_speed():
    """Estimate training throughput."""
    print("\n" + "="*80)
    print("TRAINING SPEED ESTIMATION")
    print("="*80)
    
    # Load config
    with open("config/env.yaml") as f:
        env_config = yaml.safe_load(f)
    
    with open("config/airsim_settings.json") as f:
        airsim_config = json.load(f)
    
    max_steps = env_config['env']['max_steps']
    dt = env_config['env']['sim_dt']
    num_agents = env_config['env']['num_agents']
    clock_speed = airsim_config.get('ClockSpeed', 1)
    
    episode_sim_time = max_steps * dt  # seconds in simulation
    episode_wall_time = episode_sim_time / clock_speed  # seconds in real world
    
    timesteps_per_episode = max_steps * num_agents
    timesteps_per_hour = (3600 / episode_wall_time) * timesteps_per_episode
    
    print(f"\n  Episode Configuration:")
    print(f"    Max steps: {max_steps}")
    print(f"    Timestep (dt): {dt}s")
    print(f"    Num agents: {num_agents}")
    print(f"    ClockSpeed: {clock_speed}x")
    
    print(f"\n  Episode Duration:")
    print(f"    Simulation time: {episode_sim_time:.1f}s ({episode_sim_time/60:.1f} minutes)")
    print(f"    Wall time: {episode_wall_time:.2f}s ({episode_wall_time/60:.2f} minutes)")
    print(f"    Speedup factor: {clock_speed}x")
    
    print(f"\n  Training Throughput:")
    print(f"    Timesteps per episode: {timesteps_per_episode:,}")
    print(f"    Episodes per wall hour: {3600/episode_wall_time:.1f}")
    print(f"    Timesteps per wall hour: {timesteps_per_hour:,.0f}")
    
    print(f"\n  Time to Milestones:")
    milestones = [10000, 50000, 100000, 500000, 1000000]
    for milestone in milestones:
        hours = milestone / timesteps_per_hour
        minutes = hours * 60
        if hours < 1:
            print(f"    {milestone:>8,} timesteps: {minutes:.1f} minutes")
        else:
            print(f"    {milestone:>8,} timesteps: {hours:.1f} hours ({hours/24:.1f} days)")
    
    return timesteps_per_hour

def check_modifications_applied():
    """Overall verification."""
    print("\n" + "="*80)
    print("FINAL VERIFICATION")
    print("="*80)
    
    config_ok = check_config_files()
    code_ok = check_reward_implementation()
    throughput = estimate_training_speed()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if config_ok and code_ok:
        print("\n✅ ALL MODIFICATIONS SUCCESSFULLY APPLIED!")
        print(f"\n🚀 Estimated training speed: {throughput:,.0f} timesteps/hour")
        print(f"   (That's {throughput/36000:.1f}x faster than 1x clock speed)")
        
        print("\n📋 Next Steps:")
        print("  1. Deploy settings: .\\scripts\\deploy_settings.ps1")
        print("  2. Start AirSim (Unreal Engine)")
        print("  3. Begin training: python -m training.train_rllib_ppo")
        print("  4. Monitor progress: Watch episode rewards improving")
        print("  5. Analyze results: python evaluation/analyze_movement.py")
        
        print("\n🎯 Success Criteria (check after 50k timesteps):")
        print("  - Alignment: -0.04 → >0.3")
        print("  - Progress %: -1.5% → >10%")
        print("  - Episode reward: -3000 → <-1000")
        print("  - Agents toward goal: 43% → >60%")
        
        return True
    else:
        print("\n❌ SOME MODIFICATIONS MISSING OR INCORRECT!")
        if not config_ok:
            print("  - Check config files (env.yaml, airsim_settings.json)")
        if not code_ok:
            print("  - Check utils_reward.py implementation")
        return False

if __name__ == "__main__":
    success = check_modifications_applied()
    exit(0 if success else 1)
