import os
import json
import numpy as np
import yaml
from envs.universal_uav_env import UniversalUAVEnv
from loguru import logger

def run_clean_trial(num_trials=1, max_steps=500, mode="normal"):
    print(f"🚀 Starting {'Stress' if mode=='stress' else 'Clean'} Trial (Mock Environment)...")
    
    # Initialize environment in smoke_test mode (uses MockAirSimClient)
    env = UniversalUAVEnv(config_path="config/env.yaml", smoke_test=True)
    
    # OVERRIDE: Increase max steps and disable PID assist for clean trial
    env.max_steps = max_steps
    env.use_pid_assist = False 
    
    results = []
    
    for t in range(num_trials):
        print(f"\n--- {'STRESS ' if mode=='stress' else ''}Trial {t+1} ---")
        
        # Override spawn positions for stress test
        spawn_positions = None
        if mode == "stress":
            # Spawn drones at 2m intervals (Conflict zone)
            spawn_positions = {
                f"Drone{i}": (i * 2.0, 0.0, 10.0) for i in range(env.initial_num_agents)
            }
            obs, info = env.reset() # Standard reset
            # Force teleport to stress positions
            env.airsim_client.reset(spawn_positions=spawn_positions)
            # Re-fetch states after force reset
            states = env.airsim_client.get_drone_states()
        else:
            obs, info = env.reset()
        
        agents = env.agents[:]
        goals = env.goals.copy()
        
        trial_data = {
            "trial_id": t,
            "goals": {agent: goals[agent].tolist() for agent in agents},
            "agents": {agent: {"pos": [], "actions": [], "dist_to_goal": [], "observations": []} for agent in agents},
            "success": {agent: False for agent in agents},
            "steps_to_reach": {agent: None for agent in agents}
        }
        
        # Simple Greedy Policy: Max speed towards goal
        max_speed = 5.0 # m/s
        
        for step in range(max_steps):
            # 1. Get current states
            states = env.airsim_client.get_drone_states()
            
            # Check if all reached
            all_reached = True
            actions = {}
            
            for agent in agents:
                current_pos = np.array(states[agent]['pos'])
                goal_pos = np.array(goals[agent])
                dist = np.linalg.norm(current_pos - goal_pos)
                
                # Record data
                trial_data["agents"][agent]["pos"].append(current_pos.tolist())
                trial_data["agents"][agent]["dist_to_goal"].append(float(dist))
                # obs is a dict from env.step, we need to record the value for this specific agent
                if agent in obs:
                    trial_data["agents"][agent]["observations"].append(obs[agent].tolist())
                
                if dist < 2.5: # Success radius
                    if not trial_data["success"][agent]:
                        trial_data["success"][agent] = True
                        trial_data["steps_to_reach"][agent] = step
                        print(f"  ✅ {agent} reached goal in {step} steps!")
                    
                    # Hover action
                    if env.use_simplified_actions:
                        actions[agent] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        actions[agent] = np.zeros(4, dtype=np.float32)
                else:
                    all_reached = False
                    # Calculate direction
                    direction = goal_pos - current_pos
                    dir_unit = direction / (dist + 1e-6)
                    
                    yaw = states[agent]['yaw']
                    
                    # Target bearing to goal
                    goal_bearing = np.arctan2(direction[1], direction[0])
                    yaw_error = np.arctan2(np.sin(goal_bearing - yaw), np.cos(goal_bearing - yaw))
                    yaw_rate_cmd = np.clip(yaw_error / env.dt, -1.0, 1.0) 
                    
                    # Proportional control for speed to avoid overshooting in one 3s step
                    p_speed = dist / env.dt
                    speed_cmd = min(max_speed, p_speed)
                    
                    if env.use_simplified_actions:
                        # Simplified: [speed, yaw_rate, climb_rate]
                        # We only move forward if heading error is small
                        final_speed = speed_cmd if abs(yaw_error) < 0.5 else 0.5
                        climb_rate = np.clip(direction[2] / env.dt, -2.0, 2.0)
                        act = np.array([final_speed, yaw_rate_cmd, climb_rate], dtype=np.float32)
                    else:
                        # Original: [vx_body, vy_body, vz_enu, yaw_rate]
                        # Transform v_world back to body frame
                        v_world = dir_unit * speed_cmd
                        cos_y = np.cos(yaw)
                        sin_y = np.sin(yaw)
                        vx_body = v_world[0] * cos_y + v_world[1] * sin_y
                        vy_body = -v_world[0] * sin_y + v_world[1] * cos_y
                        vz_enu = np.clip(direction[2] / env.dt, -2.0, 2.0)
                        act = np.array([vx_body, vy_body, vz_enu, yaw_rate_cmd], dtype=np.float32)
                    
                    actions[agent] = act
                    trial_data["agents"][agent]["actions"].append(act.tolist())

            # 2. Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # 3. Audit for collisions or near-misses
            for agent, info in infos.items():
                reward = rewards.get(agent, 0.0)
                R_collision_threshold = -env.config['reward'].get('R_collision', 200.0) / 2.0
                
                # Collision check (Strong: info outcome OR large negative reward)
                if info.get('outcome') in ['collision', 'stuck_collision'] or reward <= R_collision_threshold:
                    print(f"  🚨 ALERT: {agent} COLLIDED at step {step}! Total Reward: {reward:.2f}")
                    if "collisions" not in trial_data["agents"][agent]:
                        # Record position of collision
                        trial_data["agents"][agent]["collisions"] = []
                    trial_data["agents"][agent]["collisions"].append({"step": step, "pos": states[agent]['pos'], "reward": float(reward)})
                
                # Near-miss check (separation violation)
                if 'r_near' in info and info['r_near'] < 0:
                    sep = info.get('sep_min', 0.0)
                    print(f"  ⚠️ Warning: {agent} Near-miss (Sep: {sep:.2f}m) at step {step} | Reward reduction: {info['r_near']:.2f}")
                    if "near_misses" not in trial_data["agents"][agent]:
                        trial_data["agents"][agent]["near_misses"] = []
                    trial_data["agents"][agent]["near_misses"].append({"step": step, "sep": sep, "pos": states[agent]['pos'], "reward_near": float(info['r_near'])})

            if all_reached:
                print(f"  🏁 All agents reached goal at step {step}!")
                break
                
        results.append(trial_data)

    # Save results
    os.makedirs("data", exist_ok=True)
    filename = f"data/clean_trial_{mode}_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    
    # Summary
    for i, res in enumerate(results):
        success_count = sum(res["success"].values())
        print(f"Trial {i}: {success_count}/{len(res['success'])} agents reached goal.")

if __name__ == "__main__":
    # 1. Normal Clean Trial (Baseline)
    run_clean_trial(num_trials=1, mode="normal")
    
    # 2. Stress Trial (Force Near-misses / Potential Conflicts)
    run_clean_trial(num_trials=1, mode="stress")
