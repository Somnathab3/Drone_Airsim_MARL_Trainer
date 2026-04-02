"""
Analyze drone movement patterns to diagnose why drones aren't reaching goals.
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def analyze_episode(csv_path):
    """Analyze a single episode for movement patterns."""
    df = pd.read_csv(csv_path)
    
    results = {}
    
    for agent_id in df['agent_id'].unique():
        agent_df = df[df['agent_id'] == agent_id].sort_values('step')
        
        if len(agent_df) < 2:
            continue
            
        # Initial and final positions/goals
        start_pos = np.array([agent_df.iloc[0]['px_enu'], 
                              agent_df.iloc[0]['py_enu'], 
                              agent_df.iloc[0]['pz_enu']])
        end_pos = np.array([agent_df.iloc[-1]['px_enu'], 
                            agent_df.iloc[-1]['py_enu'], 
                            agent_df.iloc[-1]['pz_enu']])
        goal_pos = np.array([agent_df.iloc[0]['gx_enu'], 
                             agent_df.iloc[0]['gy_enu'], 
                             agent_df.iloc[0]['gz_enu']])
        
        # Distance metrics
        start_dist = np.linalg.norm(start_pos - goal_pos)
        end_dist = np.linalg.norm(end_pos - goal_pos)
        total_movement = np.linalg.norm(end_pos - start_pos)
        
        # Movement toward goal
        progress = start_dist - end_dist  # Positive = moved closer
        progress_percent = (progress / start_dist) * 100 if start_dist > 0 else 0
        
        # Heading analysis
        goal_direction = goal_pos - start_pos
        if np.linalg.norm(goal_direction) > 0:
            goal_direction = goal_direction / np.linalg.norm(goal_direction)
        
        # Actual movement direction
        actual_movement = end_pos - start_pos
        if np.linalg.norm(actual_movement) > 0:
            actual_movement = actual_movement / np.linalg.norm(actual_movement)
            
            # Cosine similarity (1 = perfect alignment, -1 = opposite direction, 0 = perpendicular)
            alignment = np.dot(goal_direction, actual_movement)
        else:
            alignment = 0
            
        # Velocity analysis
        speeds = np.sqrt(agent_df['vx_enu']**2 + agent_df['vy_enu']**2 + agent_df['vz_enu']**2)
        avg_speed = speeds.mean()
        max_speed = speeds.max()
        
        # Action analysis
        if 'ax_body' in agent_df.columns:
            action_magnitudes = np.sqrt(agent_df['ax_body']**2 + 
                                       agent_df['ay_body']**2 + 
                                       agent_df['az_enu']**2 + 
                                       agent_df['yaw_rate']**2)
            avg_action_mag = action_magnitudes.mean()
        else:
            avg_action_mag = 0
        
        # Reward breakdown
        total_reward = agent_df['reward'].sum()
        avg_reward = agent_df['reward'].mean()
        
        results[agent_id] = {
            'start_dist': start_dist,
            'end_dist': end_dist,
            'progress': progress,
            'progress_percent': progress_percent,
            'total_movement': total_movement,
            'alignment': alignment,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'avg_action_mag': avg_action_mag,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'steps': len(agent_df),
            'collisions': agent_df['collision'].sum(),
        }
    
    return results

def main():
    run_folder = Path("data/episodes/run_20260202_050531")
    
    if not run_folder.exists():
        print(f"Run folder not found: {run_folder}")
        return
    
    episode_files = list(run_folder.glob("episode_*.csv"))
    
    if not episode_files:
        print(f"No episode files found in {run_folder}")
        return
    
    # Sample first 5 episodes for detailed analysis
    print(f"Analyzing {len(episode_files)} episodes...\n")
    print("="*80)
    
    all_results = []
    
    # Analyze ALL episodes for statistics
    for episode_file in episode_files:
        results = analyze_episode(episode_file)
        
        for agent_id, metrics in results.items():
            all_results.append({
                'episode': episode_file.name,
                'agent': agent_id,
                **metrics
            })
    
    # Show detailed output for first 5 episodes only
    print("\nDETAILED ANALYSIS (First 5 Episodes):")
    print("="*80)
    
    for episode_file in episode_files[:5]:
        print(f"\n{episode_file.name}:")
        print("-"*80)
        
        results = analyze_episode(episode_file)
        
        for agent_id, metrics in results.items():
            print(f"  {agent_id}:")
            print(f"    Start distance: {metrics['start_dist']:.2f}m")
            print(f"    End distance: {metrics['end_dist']:.2f}m")
            print(f"    Progress: {metrics['progress']:.2f}m ({metrics['progress_percent']:.1f}%)")
            print(f"    Total movement: {metrics['total_movement']:.2f}m")
            print(f"    Alignment with goal: {metrics['alignment']:.3f} (1=perfect, 0=perpendicular, -1=opposite)")
            print(f"    Avg speed: {metrics['avg_speed']:.2f} m/s")
            print(f"    Avg action magnitude: {metrics['avg_action_mag']:.2f}")
            print(f"    Total reward: {metrics['total_reward']:.1f}")
            print(f"    Collisions: {metrics['collisions']}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    df_results = pd.DataFrame(all_results)
    
    print(f"\nProgress Analysis:")
    print(f"  Agents moving TOWARD goal: {(df_results['progress'] > 0).sum()} / {len(df_results)} ({(df_results['progress'] > 0).mean()*100:.1f}%)")
    print(f"  Agents moving AWAY from goal: {(df_results['progress'] < 0).sum()} / {len(df_results)} ({(df_results['progress'] < 0).mean()*100:.1f}%)")
    print(f"  Average progress: {df_results['progress'].mean():.2f}m")
    print(f"  Average progress %: {df_results['progress_percent'].mean():.1f}%")
    
    print(f"\nAlignment Analysis:")
    print(f"  Average alignment: {df_results['alignment'].mean():.3f}")
    print(f"  Agents moving in goal direction (>0.5): {(df_results['alignment'] > 0.5).sum()} / {len(df_results)} ({(df_results['alignment'] > 0.5).mean()*100:.1f}%)")
    print(f"  Agents moving perpendicular ([-0.5, 0.5]): {((df_results['alignment'] >= -0.5) & (df_results['alignment'] <= 0.5)).sum()} / {len(df_results)}")
    print(f"  Agents moving opposite (<-0.5): {(df_results['alignment'] < -0.5).sum()} / {len(df_results)} ({(df_results['alignment'] < -0.5).mean()*100:.1f}%)")
    
    print(f"\nMovement Analysis:")
    print(f"  Average speed: {df_results['avg_speed'].mean():.2f} m/s")
    print(f"  Speed range: [{df_results['avg_speed'].min():.2f}, {df_results['avg_speed'].max():.2f}] m/s")
    print(f"  Average action magnitude: {df_results['avg_action_mag'].mean():.2f}")
    print(f"  Action magnitude range: [{df_results['avg_action_mag'].min():.2f}, {df_results['avg_action_mag'].max():.2f}]")
    print(f"  Average total movement: {df_results['total_movement'].mean():.2f}m")
    print(f"  Total movement range: [{df_results['total_movement'].min():.2f}, {df_results['total_movement'].max():.2f}]m")
    
    print(f"\nReward Analysis:")
    print(f"  Average total reward: {df_results['total_reward'].mean():.1f}")
    print(f"  Total reward range: [{df_results['total_reward'].min():.1f}, {df_results['total_reward'].max():.1f}]")
    print(f"  Average step reward: {df_results['avg_reward'].mean():.2f}")
    print(f"  Step reward range: [{df_results['avg_reward'].min():.2f}, {df_results['avg_reward'].max():.2f}]")
    
    print(f"\nDistance Analysis:")
    print(f"  Average start distance: {df_results['start_dist'].mean():.2f}m")
    print(f"  Start distance range: [{df_results['start_dist'].min():.2f}, {df_results['start_dist'].max():.2f}]m")
    print(f"  Average end distance: {df_results['end_dist'].mean():.2f}m")
    print(f"  End distance range: [{df_results['end_dist'].min():.2f}, {df_results['end_dist'].max():.2f}]m")
    print(f"  Average progress: {df_results['progress'].mean():.2f}m")
    print(f"  Progress range: [{df_results['progress'].min():.2f}, {df_results['progress'].max():.2f}]m")
    
    print(f"\nCollision Analysis:")
    print(f"  Average collisions per agent: {df_results['collisions'].mean():.1f}")
    print(f"  Collision range: [{df_results['collisions'].min()}, {df_results['collisions'].max()}]")
    print(f"  Total agents analyzed: {len(df_results)}")
    print(f"  Total episodes analyzed: {df_results['episode'].nunique()}")
    
    print(f"\nDiagnosis:")
    print("="*80)
    
    # Key issues
    if df_results['alignment'].mean() < 0.3:
        print("❌ CRITICAL: Drones are NOT moving toward their goals!")
        print("   Average alignment is {:.3f} (should be > 0.7 for goal-directed behavior)".format(df_results['alignment'].mean()))
    
    if df_results['progress_percent'].mean() < 10:
        print("❌ CRITICAL: Drones making minimal progress!")
        print("   Average progress is only {:.1f}% of initial distance".format(df_results['progress_percent'].mean()))
    
    if df_results['avg_speed'].mean() < 1.0:
        print("❌ WARNING: Drones moving very slowly!")
        print("   Average speed {:.2f} m/s (action space allows up to 10 m/s)".format(df_results['avg_speed'].mean()))
    
    if df_results['avg_action_mag'].mean() < 2.0:
        print("❌ WARNING: Actions are very small!")
        print("   Average action magnitude {:.2f} (max possible ~12.5)".format(df_results['avg_action_mag'].mean()))
        print("   Possible cause: Action cost penalty too high, or policy not exploring")

if __name__ == "__main__":
    main()
