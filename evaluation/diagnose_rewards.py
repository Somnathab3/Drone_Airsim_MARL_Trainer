"""
Diagnose the catastrophic reward issue from training.
Episode rewards of -18000 indicate something is very wrong.
"""
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def diagnose_training(episodes_dir="data/episodes", num_episodes=20):
    """Diagnose what's causing the catastrophic rewards."""
    
    episode_files = sorted(glob.glob(os.path.join(episodes_dir, "episode_*.csv")))
    
    if not episode_files:
        print(f"No episode files found in {episodes_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC ANALYSIS - Last {num_episodes} Episodes")
    print(f"Episode Rewards from Training: -18000 to -17800")
    print(f"This indicates CATASTROPHIC failure - analyzing root cause...")
    print(f"{'='*80}\n")
    
    # Analyze last N episodes
    total_collisions = 0
    total_steps = 0
    total_cumulative_reward = 0
    total_near_misses = 0
    episodes_analyzed = 0
    
    for filepath in episode_files[-num_episodes:]:
        try:
            df = pd.read_csv(filepath)
            episode_name = Path(filepath).stem
            
            # Get episode stats
            num_steps = len(df)
            total_steps += num_steps
            
            # Get unique agents
            agents = df['agent_id'].unique()
            
            print(f"\n--- {episode_name} ({num_steps} steps, {len(agents)} agents) ---")
            
            # Per-agent analysis
            for agent in agents:
                agent_df = df[df['agent_id'] == agent]
                
                # Reward analysis
                total_reward = agent_df['reward'].sum()
                mean_reward = agent_df['reward'].mean()
                min_reward = agent_df['reward'].min()
                
                total_cumulative_reward += total_reward
                
                # Collision analysis
                if 'collision' in agent_df.columns:
                    num_collisions = agent_df['collision'].sum()
                    total_collisions += num_collisions
                else:
                    num_collisions = 0
                
                # Near-miss analysis
                if 'sep_min' in agent_df.columns:
                    min_sep = agent_df['sep_min'].min()
                    near_misses = (agent_df['sep_min'] < 3.0).sum()  # d_safe = 3.0
                    total_near_misses += near_misses
                else:
                    min_sep = "N/A"
                    near_misses = 0
                
                # Outcome
                if 'outcome' in agent_df.columns:
                    final_outcome = agent_df['outcome'].iloc[-1]
                else:
                    final_outcome = "unknown"
                
                # Distance to goal analysis
                if 'gx_enu' in agent_df.columns and 'px_enu' in agent_df.columns:
                    final_pos = np.array([
                        agent_df['px_enu'].iloc[-1],
                        agent_df['py_enu'].iloc[-1],
                        agent_df['pz_enu'].iloc[-1]
                    ])
                    goal_pos = np.array([
                        agent_df['gx_enu'].iloc[-1],
                        agent_df['gy_enu'].iloc[-1],
                        agent_df['gz_enu'].iloc[-1]
                    ])
                    final_dist = np.linalg.norm(final_pos - goal_pos)
                else:
                    final_dist = "N/A"
                
                print(f"  {agent}:")
                print(f"    Total Reward: {total_reward:,.0f}")
                print(f"    Mean Step Reward: {mean_reward:.2f}")
                print(f"    Min Step Reward: {min_reward:.2f}")
                print(f"    Collisions: {num_collisions}")
                print(f"    Near-Misses (sep<3m): {near_misses} steps")
                print(f"    Min Separation: {min_sep}")
                print(f"    Final Dist to Goal: {final_dist:.2f}m" if isinstance(final_dist, float) else f"    Final Dist: {final_dist}")
                print(f"    Outcome: {final_outcome}")
            
            episodes_analyzed += 1
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY ({episodes_analyzed} episodes analyzed)")
    print(f"{'='*80}")
    print(f"  Total Steps: {total_steps:,}")
    print(f"  Total Collisions: {total_collisions}")
    print(f"  Total Near-Misses: {total_near_misses}")
    print(f"  Avg Cumulative Reward/Agent: {total_cumulative_reward / (episodes_analyzed * 5):.0f}")
    print(f"  Collision Rate: {total_collisions / total_steps * 100:.2f}% of steps")
    print(f"  Near-Miss Rate: {total_near_misses / total_steps * 100:.2f}% of steps")
    
    # Diagnosis
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS")
    print(f"{'='*80}")
    
    avg_reward_per_agent = total_cumulative_reward / (episodes_analyzed * 5)
    
    if avg_reward_per_agent < -10000:
        print("  ⚠️  CRITICAL: Catastrophic rewards detected!")
        print(f"     Expected episode reward: ~0 to -1000")
        print(f"     Actual: {avg_reward_per_agent:.0f}")
        print()
        print("  Likely causes:")
        print("  1. Continuous collisions causing -500 penalty every step")
        print("  2. Near-miss penalty (-beta * (3-sep)^2) being applied too frequently")
        print("  3. Drones not making progress toward goals (negative progress)")
        print("  4. Time penalty accumulating without goal reaching")
        print()
        print("  Recommended fixes:")
        print("  - Reduce collision penalty (500 → 100)")
        print("  - Reduce near-miss beta coefficient (10 → 1)")
        print("  - Increase goal radius to make success easier")
        print("  - Add positive shaping for velocity toward goal")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    diagnose_training(num_episodes=20)
