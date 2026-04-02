import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

def plot_run_metrics(run_dir):
    print(f"📊 Generating Enhanced Dashboard for: {run_dir}")
    
    csv_files = glob.glob(os.path.join(run_dir, "episode_*.csv"))
    if not csv_files:
        print(f"❌ No episode CSV files found in {run_dir}")
        return

    # Sort episodes chronologically
    csv_files.sort(key=os.path.getmtime)
    
    episode_data = []
    
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            if df.empty: continue
                
            # 1. Total Reward
            total_reward = df['reward'].sum()
            
            # 2. Collision Count
            if df['collision'].dtype == 'bool':
                collisions = df['collision'].sum()
            else:
                collisions = df['collision'].astype(str).str.lower().str.contains('true').sum()
            
            # 3. Success Rate
            # unique agents that reached goal / total unique agents
            total_agents = df['agent_id'].nunique()
            if 'goal_reached' in df.columns:
                if df['goal_reached'].dtype == 'bool':
                    reached_agents = df[df['goal_reached'] == True]['agent_id'].nunique()
                else:
                    reached_agents = df[df['goal_reached'].astype(str).str.lower().str.contains('true')]['agent_id'].nunique()
                success_rate = (reached_agents / total_agents) * 100
            else:
                success_rate = 0.0
                
            # 4. Min Separation (Safety Margin)
            min_sep = df['sep_min'].min() if 'sep_min' in df.columns else 0.0
            
            # 5. Episode Length
            ep_length = df['step'].max()
            
            # 6. Avg Speed
            if all(c in df.columns for c in ['vx_enu', 'vy_enu', 'vz_enu']):
                speeds = np.sqrt(df['vx_enu']**2 + df['vy_enu']**2 + df['vz_enu']**2)
                avg_speed = speeds.mean()
            else:
                avg_speed = 0.0
                
            episode_id = os.path.basename(file_path).replace('episode_', '').replace('.csv', '')
            
            episode_data.append({
                'episode_idx': i + 1,
                'total_reward': total_reward,
                'collisions': int(collisions),
                'success_rate': success_rate,
                'min_sep': min_sep,
                'ep_length': ep_length,
                'avg_speed': avg_speed
            })
        except Exception as e:
            print(f"⚠️ Warning: Error in {os.path.basename(file_path)}: {e}")

    if not episode_data:
        print("❌ No valid data extracted.")
        return

    metrics_df = pd.DataFrame(episode_data)
    
    # Dashboard Layout (3x2)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f"Multi-Agent UAV Training Dashboard: {os.path.basename(run_dir)}", fontsize=20, y=1.02)
    
    plot_configs = [
        ('total_reward', 'Total Reward', 'b', axes[0, 0]),
        ('success_rate', 'Success Rate (%)', 'g', axes[0, 1]),
        ('collisions', 'Collision Count', 'r', axes[1, 0]),
        ('min_sep', 'Min Separation (m)', 'purple', axes[1, 1]),
        ('ep_length', 'Episode Length (Steps)', 'orange', axes[2, 0]),
        ('avg_speed', 'Avg Speed (m/s)', 'brown', axes[2, 1])
    ]
    
    for col, title, color, ax in plot_configs:
        # Plot raw data
        ax.plot(metrics_df['episode_idx'], metrics_df[col], marker='o', alpha=0.3, color=color, linestyle='-')
        
        # Plot rolling mean (window=max(2, len/5))
        window = max(2, len(metrics_df)//5)
        rolling = metrics_df[col].rolling(window=window, min_periods=1).mean()
        ax.plot(metrics_df['episode_idx'], rolling, color=color, linewidth=3, label=f'Trend (W={window})')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode Index')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        # Specific formatting
        if col == 'success_rate': ax.set_ylim(-5, 105)
        if col == 'collisions': ax.set_ylim(-0.5, metrics_df['collisions'].max() + 1)
        if col == 'min_sep': 
            ax.axhline(y=2.0, color='red', linestyle='--', label='Safety Limit (2m)')
            ax.legend()

    plt.tight_layout()
    
    output_path = os.path.join(run_dir, "run_analysis.png")
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Dashboard generated: {output_path}")
    
    # Console Summary
    print(f"\nFinal Stats over {len(episode_data)} Episodes:")
    print(f"  - Avg Success Rate: {metrics_df['success_rate'].mean():.1f}%")
    print(f"  - Total Collisions: {metrics_df['collisions'].sum()}")
    print(f"  - Global Min Separation: {metrics_df['min_sep'].min():.2f}m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=str)
    args = parser.parse_args()
    plot_run_metrics(args.run_dir)
