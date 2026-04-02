import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_clean_trial(json_path="data/clean_trial_results.json"):
    print(f"📊 Visualizing trial from {json_path}...")
    
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found.")
        return
        
    with open(json_path, "r") as f:
        trials = json.load(f)
        
    for i, trial in enumerate(trials):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        
        goals = trial["goals"]
        
        for j, (agent, data) in enumerate(trial["agents"].items()):
            color = colors[j % len(colors)]
            pos = np.array(data["pos"])
            goal = np.array(goals[agent])
            
            if len(pos) == 0:
                continue
                
            # Plot Trajectory
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"{agent} Path", color=color, linewidth=2, alpha=0.7)
            
            # Start Point
            ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, s=50, marker='o', label=f"{agent} Start" if i==0 else "")
            
            # Goal Point
            ax.scatter(goal[0], goal[1], goal[2], color=color, s=100, marker='X', edgecolors='black', label=f"{agent} Goal" if i==0 else "")
            
            # Connecting line to goal from last point (if not reached)
            ax.plot([pos[-1, 0], goal[0]], [pos[-1, 1], goal[1]], [pos[-1, 2], goal[2]], color=color, linestyle='--', alpha=0.3)

        ax.set_xlabel('X (ENU)')
        ax.set_ylabel('Y (ENU)')
        ax.set_zlabel('Z (Altitude)')
        ax.set_title(f"Clean Trial {trial['trial_id']} - UAV Trajectories (No Collisions)")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        output_path = f"data/clean_trial_t{trial['trial_id']}_viz.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"✅ Visualization saved to {output_path}")
        
    plt.close('all')

if __name__ == "__main__":
    visualize_clean_trial()
