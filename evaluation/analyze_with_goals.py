"""
Analyze episodes with goal positions, speeds, and collision context.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_episode_with_goals(csv_path):
    """Analyze single episode with detailed goal and collision info."""
    df = pd.read_csv(csv_path)
    
    results = []
    agents = df['agent_id'].unique()
    
    for agent in agents:
        agent_df = df[df['agent_id'] == agent].sort_values('step')
        
        if len(agent_df) == 0:
            continue
        
        # Get positions
        start = agent_df.iloc[0]
        end = agent_df.iloc[-1]
        
        # Goal position (constant throughout episode)
        goal_pos = np.array([start['gx_enu'], start['gy_enu'], start['gz_enu']])
        
        # Start/end positions
        start_pos = np.array([start['px_enu'], start['py_enu'], start['pz_enu']])
        end_pos = np.array([end['px_enu'], end['py_enu'], end['pz_enu']])
        
        # Distances
        start_dist = np.linalg.norm(goal_pos - start_pos)
        end_dist = np.linalg.norm(goal_pos - end_pos)
        progress_m = start_dist - end_dist
        progress_pct = (progress_m / start_dist) * 100 if start_dist > 0 else 0
        
        # Movement metrics
        speeds = np.sqrt(agent_df['vx_enu']**2 + agent_df['vy_enu']**2 + agent_df['vz_enu']**2)
        avg_speed = speeds.mean()
        max_speed = speeds.max()
        
        # Collision analysis
        collisions = agent_df['collision'].sum()
        total_steps = len(agent_df)
        collision_rate = (collisions / total_steps) * 100 if total_steps > 0 else 0
        
        # Separation (proximity to obstacles/other agents)
        min_separation = agent_df['sep_min'].min()
        avg_separation = agent_df['sep_min'].mean()
        
        # Reward breakdown
        total_reward = agent_df['reward'].sum()
        avg_step_reward = agent_df['reward'].mean()
        
        results.append({
            'agent': agent,
            'goal_pos': goal_pos,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'start_dist': start_dist,
            'end_dist': end_dist,
            'progress_m': progress_m,
            'progress_pct': progress_pct,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'collisions': collisions,
            'total_steps': total_steps,
            'collision_rate': collision_rate,
            'min_separation': min_separation,
            'avg_separation': avg_separation,
            'total_reward': total_reward,
            'avg_step_reward': avg_step_reward
        })
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_with_goals.py <episode_csv_path>")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"EPISODE ANALYSIS: {csv_path.name}")
    print(f"{'='*80}\n")
    
    results = analyze_episode_with_goals(csv_path)
    
    for r in results:
        print(f"{r['agent']}:")
        print(f"  Goal position: [{r['goal_pos'][0]:.1f}, {r['goal_pos'][1]:.1f}, {r['goal_pos'][2]:.1f}]m")
        print(f"  Start position: [{r['start_pos'][0]:.1f}, {r['start_pos'][1]:.1f}, {r['start_pos'][2]:.1f}]m")
        print(f"  End position: [{r['end_pos'][0]:.1f}, {r['end_pos'][1]:.1f}, {r['end_pos'][2]:.1f}]m")
        print(f"  Initial distance to goal: {r['start_dist']:.2f}m")
        print(f"  Final distance to goal: {r['end_dist']:.2f}m")
        print(f"  Progress: {r['progress_m']:.2f}m ({r['progress_pct']:.1f}%)")
        print(f"  Speed: avg={r['avg_speed']:.2f} m/s, max={r['max_speed']:.2f} m/s")
        print(f"  Collisions: {r['collisions']} ({r['collision_rate']:.1f}% of {r['total_steps']} steps)")
        print(f"  Separation: min={r['min_separation']:.2f}m, avg={r['avg_separation']:.2f}m")
        print(f"  Reward: total={r['total_reward']:.1f}, avg/step={r['avg_step_reward']:.2f}")
        print()
    
    # Summary statistics
    avg_progress_pct = np.mean([r['progress_pct'] for r in results])
    avg_speed = np.mean([r['avg_speed'] for r in results])
    max_speed_overall = np.max([r['max_speed'] for r in results])
    avg_collisions = np.mean([r['collisions'] for r in results])
    agents_reached_goal = sum(1 for r in results if r['end_dist'] < 5.0)  # Within goal_radius=5m
    
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Agents analyzed: {len(results)}")
    print(f"Average progress: {avg_progress_pct:.1f}%")
    print(f"Average speed: {avg_speed:.2f} m/s")
    print(f"Max speed achieved: {max_speed_overall:.2f} m/s")
    print(f"Average collisions per agent: {avg_collisions:.1f}")
    print(f"Agents that reached goal (<5m): {agents_reached_goal}/{len(results)}")
    print()

if __name__ == "__main__":
    main()
