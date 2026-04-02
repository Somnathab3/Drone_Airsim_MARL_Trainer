"""
Quick analysis script for episode logs.
Prints summary statistics to understand what's happening during training.
"""
import pandas as pd
import glob
import os
from pathlib import Path

def analyze_episodes(episodes_dir="data/episodes"):
    """Analyze all episode CSV files and print summary stats."""
    
    episode_files = glob.glob(os.path.join(episodes_dir, "episode_*.csv"))
    
    if not episode_files:
        print(f"No episode files found in {episodes_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Found {len(episode_files)} episode files")
    print(f"{'='*70}\n")
    
    # Analyze each episode
    for i, filepath in enumerate(sorted(episode_files)[-10:]):  # Last 10 episodes
        try:
            df = pd.read_csv(filepath)
            episode_name = Path(filepath).stem
            
            print(f"\n--- {episode_name} ---")
            print(f"  Steps: {len(df)}")
            
            # Get unique agents
            agents = df['agent'].unique()
            print(f"  Agents: {len(agents)}")
            
            for agent in agents:
                agent_df = df[df['agent'] == agent]
                
                # Reward statistics
                total_reward = agent_df['reward'].sum()
                min_reward = agent_df['reward'].min()
                max_reward = agent_df['reward'].max()
                
                # Cumulative reward (last value)
                if 'cumulative_reward' in agent_df.columns:
                    final_cumulative = agent_df['cumulative_reward'].iloc[-1]
                else:
                    final_cumulative = total_reward
                
                # Collisions
                collisions = agent_df['collision'].sum() if 'collision' in agent_df.columns else 0
                
                # Final outcome
                if 'outcome' in agent_df.columns:
                    final_outcome = agent_df['outcome'].iloc[-1]
                else:
                    final_outcome = "unknown"
                
                # Min separation
                if 'sep_min' in agent_df.columns:
                    min_sep = agent_df['sep_min'].min()
                else:
                    min_sep = "N/A"
                
                print(f"    {agent}:")
                print(f"      Cumulative Reward: {final_cumulative:.1f}")
                print(f"      Step Rewards: min={min_reward:.2f}, max={max_reward:.2f}")
                print(f"      Collisions: {collisions}")
                print(f"      Min Separation: {min_sep}")
                print(f"      Final Outcome: {final_outcome}")
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    analyze_episodes()
