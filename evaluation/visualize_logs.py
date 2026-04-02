import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import os

def plot_trajectories(log_dir):
    files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not files:
        print(f"No CSV logs found in {log_dir}")
        return
    
    print(f"Found {len(files)} episodes.")
    
    # Plot last episode
    latest_file = max(files, key=os.path.getctime)
    print(f"Plotting latest: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Filter by agent
    agents = df['agent_id'].unique()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for agent in agents:
        data = df[df['agent_id'] == agent]
        ax.plot(data['px_enu'], data['py_enu'], data['pz_enu'], label=f"{agent}")
        
        # Plot goal
        gx = data['gx_enu'].iloc[0]
        gy = data['gy_enu'].iloc[0]
        gz = data['gz_enu'].iloc[0]
        ax.scatter(gx, gy, gz, marker='x', label=f"{agent} Goal")
        
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.legend()
    plt.title(f"Trajectory: {os.path.basename(latest_file)}")
    
    output_path = latest_file.replace(".csv", ".png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    # plt.show() # blocking

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="data/episodes")
    args = parser.parse_args()
    
    plot_trajectories(args.log_dir)
