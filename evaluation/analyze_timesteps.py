"""
Analyze if max_steps is sufficient for drones to reach goals.
"""
import numpy as np

# Current configuration
max_steps = 1000
dt = 0.1  # seconds per step
total_time = max_steps * dt  # 100 seconds

# Spawn positions (grid)
spawn_min = (0, 0, 2)      # First drone
spawn_max = (20, 10, 2)    # Last drone (5 drones in 3x2 grid, spacing=10m)

# Goal positions (random)
goal_min = (-50, -50, 5)
goal_max = (50, 50, 15)

# Max velocities from action space
max_vx = 10.0  # m/s forward
max_vy = 5.0   # m/s lateral
max_vz = 5.0   # m/s vertical

# Calculate worst-case distance
spawn_center = np.array([(spawn_min[i] + spawn_max[i])/2 for i in range(3)])
goal_corner = np.array(goal_max)
worst_case_distance = np.linalg.norm(goal_corner - spawn_min)

print("="*80)
print("TIMESTEP FEASIBILITY ANALYSIS")
print("="*80)

print(f"\n--- Configuration ---")
print(f"max_steps: {max_steps}")
print(f"dt: {dt} seconds")
print(f"Total time available: {total_time} seconds ({total_time/60:.1f} minutes)")

print(f"\n--- Spawn Area ---")
print(f"Min: {spawn_min}")
print(f"Max: {spawn_max}")

print(f"\n--- Goal Area ---")
print(f"Min: {goal_min}")
print(f"Max: {goal_max}")

print(f"\n--- Action Bounds (Max Velocities) ---")
print(f"vx_body (forward): ±{max_vx} m/s")
print(f"vy_body (lateral): ±{max_vy} m/s")
print(f"vz_enu (vertical): ±{max_vz} m/s")
print(f"Max speed (if all axes): {np.sqrt(max_vx**2 + max_vy**2 + max_vz**2):.2f} m/s")

print(f"\n--- Distance Analysis ---")

# Typical distances
spawn_to_goal_avg = []
for _ in range(1000):
    spawn_x = np.random.uniform(spawn_min[0], spawn_max[0])
    spawn_y = np.random.uniform(spawn_min[1], spawn_max[1])
    spawn_z = spawn_min[2]
    spawn = np.array([spawn_x, spawn_y, spawn_z])
    
    goal_x = np.random.uniform(goal_min[0], goal_max[0])
    goal_y = np.random.uniform(goal_min[1], goal_max[1])
    goal_z = np.random.uniform(goal_min[2], goal_max[2])
    goal = np.array([goal_x, goal_y, goal_z])
    
    dist = np.linalg.norm(goal - spawn)
    spawn_to_goal_avg.append(dist)

avg_distance = np.mean(spawn_to_goal_avg)
max_distance = np.max(spawn_to_goal_avg)
min_distance = np.min(spawn_to_goal_avg)

print(f"Average spawn-to-goal distance: {avg_distance:.2f} m")
print(f"Max spawn-to-goal distance: {max_distance:.2f} m")
print(f"Min spawn-to-goal distance: {min_distance:.2f} m")

print(f"\n--- Time Requirements ---")

# Straight-line flight at max forward speed
time_avg_max_speed = avg_distance / max_vx
time_max_distance_max_speed = max_distance / max_vx

print(f"\nBest case (straight line at max vx={max_vx} m/s):")
print(f"  Average distance: {time_avg_max_speed:.1f} seconds ({time_avg_max_speed/60:.2f} min)")
print(f"  Max distance: {time_max_distance_max_speed:.1f} seconds ({time_max_distance_max_speed/60:.2f} min)")

# Realistic flight (80% efficiency, accounting for acceleration, maneuvering)
efficiency = 0.6  # Conservative: account for collision avoidance, not straight line
realistic_speed = max_vx * efficiency
time_avg_realistic = avg_distance / realistic_speed
time_max_realistic = max_distance / realistic_speed

print(f"\nRealistic case ({efficiency*100:.0f}% efficiency at {realistic_speed:.1f} m/s avg):")
print(f"  Average distance: {time_avg_realistic:.1f} seconds ({time_avg_realistic/60:.2f} min)")
print(f"  Max distance: {time_max_realistic:.1f} seconds ({time_max_realistic/60:.2f} min)")

# Add safety margin
safety_margin = 1.5  # 50% extra time
time_with_margin = time_max_realistic * safety_margin

print(f"\nWith {safety_margin}x safety margin:")
print(f"  Required time: {time_with_margin:.1f} seconds ({time_with_margin/60:.2f} min)")

print(f"\n--- Verdict ---")
print(f"Available time: {total_time} seconds")
print(f"Required time (avg, realistic): {time_avg_realistic:.1f} seconds")
print(f"Required time (max, with margin): {time_with_margin:.1f} seconds")

if total_time >= time_with_margin:
    print(f"✅ SUFFICIENT: {total_time} >= {time_with_margin:.1f} seconds")
    print(f"   Margin: {total_time - time_with_margin:.1f} seconds ({(total_time/time_with_margin - 1)*100:.0f}% extra)")
else:
    print(f"⚠️  INSUFFICIENT: {total_time} < {time_with_margin:.1f} seconds")
    print(f"   Deficit: {time_with_margin - total_time:.1f} seconds")
    
    # Recommended max_steps
    recommended_steps = int(np.ceil(time_with_margin / dt))
    print(f"\n📊 RECOMMENDATION:")
    print(f"   Current max_steps: {max_steps}")
    print(f"   Recommended max_steps: {recommended_steps}")
    print(f"   New total time: {recommended_steps * dt:.1f} seconds ({recommended_steps * dt / 60:.1f} minutes)")

print("\n" + "="*80)

# Check current episode data (if exists)
try:
    import glob
    import pandas as pd

    episode_files = glob.glob("data/episodes/episode_*.csv")
    if episode_files:
        print("\n--- Analysis of Actual Episodes ---")
        
        total_episodes = 0
        reached_max_steps = 0
        avg_steps = []
        
        for file in episode_files[:50]:  # Sample first 50
            try:
                df = pd.read_csv(file)
                steps = len(df) // 5  # Divide by num agents
                avg_steps.append(steps)
                if steps >= max_steps:
                    reached_max_steps += 1
                total_episodes += 1
            except:
                pass
        
        if total_episodes > 0:
            print(f"Analyzed {total_episodes} episodes:")
            print(f"  Episodes hitting max_steps: {reached_max_steps}/{total_episodes} ({reached_max_steps/total_episodes*100:.1f}%)")
            print(f"  Average episode length: {np.mean(avg_steps):.0f} steps")
            print(f"  Max episode length seen: {np.max(avg_steps):.0f} steps")
            
            if reached_max_steps / total_episodes > 0.5:
                print(f"\n⚠️  > 50% of episodes hit max_steps limit!")
                print(f"   This suggests the time limit is too restrictive.")
            else:
                print(f"\n✅ Most episodes finish before max_steps.")
except Exception as e:
    print(f"\nNote: Could not analyze episodes: {e}")

print("\n" + "="*80)
