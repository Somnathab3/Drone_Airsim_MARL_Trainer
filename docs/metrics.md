# Metrics

Key metrics tracked during training and evaluation:

## Episode-Level Performance
* **Success Rate**: Fraction of agents reaching the goal without collision per episode.
* **Collision Rate**: Fraction of agents colliding per episode.
* **Goal Reached**: Boolean per agent indicating goal achievement.
* **Episode Length**: Number of timesteps before termination.
* **Cumulative Reward**: Total reward accumulated per agent.

## Swarm Coordination Metrics
* **Min Separation**: Minimum distance observed between any two agents in the swarm.
* **Average Separation**: Mean inter-agent distance across all pairs.
* **Cohesion Distance**: Average distance of agents from swarm centroid.
* **Velocity Alignment**: Average cosine similarity of agent velocities (1.0 = perfect alignment).
* **Time-to-Collision (TTC)**: Minimum predicted collision time across all agent pairs.

## Reward Components (Per Step)
* `r_progress`: Progress toward goal.
* `r_goal`: Goal achievement bonus (+100).
* `r_collision`: Collision penalty (-50).
* `r_near_miss`: Safe separation penalty.
* `r_time`: Time penalty (-0.1).
* `r_cohesion`: Swarm cohesion reward.
* `r_velocity_align`: Velocity alignment reward.
* `r_ttc`: Time-to-collision safety reward.
* `r_action_cost`: Action magnitude penalty.
* `r_smoothness`: Action smoothness penalty.
* `r_heading`: Heading alignment reward.
* `r_altitude`: Altitude control reward.

## Control System Metrics
* **Action Magnitude**: L2 norm of RL policy output.
* **Cumulative Action**: Sum of actions over accumulation period.
* **PID Contribution**: Magnitude of PID stabilization commands.
* **Action Frequency**: Steps between AirSim commands (default: 10).

## RLlib / TensorBoard
Standard PPO metrics are logged to `models/logs/`:
* `episode_reward_mean`: Average total reward per episode.
* `episode_len_mean`: Average episode length.
* `custom_metrics/success_rate_mean`: Goal achievement rate.
* `custom_metrics/collision_rate_mean`: Collision frequency.
* `custom_metrics/min_separation_min`: Closest approach distance.
* `custom_metrics/cohesion_mean`: Average swarm cohesion.
* `custom_metrics/velocity_alignment_mean`: Swarm coordination measure.

## Episode Logs
Detailed CSV logs saved to `data/episodes/run_YYYYMMDD_HHMMSS/`:

**Columns**:
- `time`: Unix timestamp
- `step`: Episode step number
- `agent_id`: Drone identifier (Drone0, Drone1, ...)
- `px_enu, py_enu, pz_enu`: Position in ENU frame (meters)
- `vx_enu, vy_enu, vz_enu`: Velocity in ENU frame (m/s)
- `yaw_enu`: Heading angle (radians)
- `gx_enu, gy_enu, gz_enu`: Goal position (meters)
- `ax_body, ay_body, az_enu, yaw_rate`: Action sent to AirSim
- `reward`: Total reward for this step
- `collision`: Boolean collision flag
- `goal_reached`: Boolean goal achievement flag
- `sep_min`: Minimum separation to other agents (meters)

## Analysis Tools

**Analyze Episodes**:
```bash
python evaluation/analyze_episodes.py --run run_20260202_141049
# Summary statistics, collision analysis, success rates
```

**Analyze Movement**:
```bash
python evaluation/analyze_movement.py --episode episode_abc123.csv
# Trajectory analysis, velocity profiles, acceleration patterns
```

**Diagnose Rewards**:
```bash
python evaluation/diagnose_rewards.py --run run_20260202_141049
# Reward component breakdown, distribution analysis
```

**Visualize Training**:
```bash
tensorboard --logdir models/logs
# Real-time training progress, metric trends
```
