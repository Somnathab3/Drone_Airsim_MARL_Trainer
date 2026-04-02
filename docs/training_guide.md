# Training System Overview

## Architecture

The Universal UAV RL training system uses a hybrid approach combining:
1. **PPO (Proximal Policy Optimization)** - RLlib implementation
2. **PID-Assisted Stabilization** - Prevents falling during early training
3. **Cumulative Action Control** - Smoother commands via action accumulation
4. **Swarm Coordination Rewards** - Multi-agent cooperation incentives

## Key Features

### 1. Egocentric Body-Frame Observations (50-dim)
- All spatial features in drone's local coordinate frame
- Better generalization and sample efficiency
- See [obs_definition.md](obs_definition.md) for details

### 2. Multi-Component Reward Function
- **Goal-seeking**: Progress rewards, goal bonus
- **Safety**: Collision penalties, near-miss penalties, TTC-based avoidance
- **Swarm**: Cohesion, velocity alignment
- **Efficiency**: Action costs, smoothness penalties
- See [reward_design.md](reward_design.md) for full breakdown

### 3. PID-Assisted Training
- **30% PID** stabilization + **70% RL** policy
- Prevents catastrophic falling during exploration
- Gradually fades as policy improves
- Configurable blend weight: `pid_blend_weight` in `config/env.yaml`

### 4. Cumulative Action System
- RL outputs action every step
- Actions accumulate over 10 steps (default)
- Accumulated sum sent to AirSim every 10th step
- Results in smoother flight and less communication overhead
- Configurable: `action_frequency` in `config/env.yaml`

## Training Configuration

### Environment Settings (`config/env.yaml`)

```yaml
env:
  num_agents: 5                    # Swarm size
  max_steps: 1000                  # Episode timeout
  sim_dt: 0.1                      # Simulation timestep (seconds)
  curriculum_2d_mode: false        # Lock Z-axis for 2D training
  use_pid_assist: true             # Enable PID stabilization
  pid_blend_weight: 0.3            # 30% PID, 70% RL
  action_frequency: 10             # Send actions every N steps
```

### Training Hyperparameters (`config/training.yaml`)

```yaml
training:
  train_batch_size: 8000           # Increased for swarm variance
  gamma: 0.995                     # Discount factor (long horizon)
  lr: 0.0001                       # Learning rate (stable for AirSim)
  sgd_minibatch_size: 128
  num_sgd_iter: 10
  lambda: 0.95                     # GAE lambda
  clip_param: 0.2                  # PPO clip ratio
  grad_clip: 0.5                   # Gradient clipping (stability)
  vf_loss_coeff: 1.0
  entropy_coeff: 0.01
  
runners:
  num_env_runners: 0               # Use local worker only (single AirSim)
```

### Reward Weights (`config/env.yaml`)

```yaml
reward:
  R_goal: 100.0                    # Goal bonus
  R_collision: 50.0                # Collision penalty
  alpha: 20.0                      # Progress coefficient
  beta: 0.5                        # Near-miss penalty
  eta: 0.1                         # Time penalty
  gamma: 0.005                     # Action cost
  lambda: 0.01                     # Action smoothness
  zeta: 1.5                        # Heading reward
  omega: 0.3                       # Altitude control
  d_safe: 5.0                      # Safe separation distance
  goal_radius: 5.0                 # Goal achievement radius
  
  # Swarm coordination
  w_cohesion: 0.1                  # Cohesion weight
  w_velocity_align: 0.05           # Velocity alignment weight
  w_ttc: 0.2                       # Time-to-collision weight
  r_cohesion_max: 20.0             # Max distance from centroid
  ttc_tau: 2.0                     # TTC time constant
```

## Quick Start

### 1. Setup Virtual Environment
```bash
cd universal-uav-rl
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure AirSim
```bash
# Copy settings to AirSim directory
cp config/airsim_settings.json "$env:USERPROFILE\Documents\AirSim\settings.json"
```

### 4. Start AirSim
- Launch Unreal Engine with Blocks environment
- Wait for "Press any key to continue" message
- Press a key to start

### 5. Run Training
```powershell
cd f:\Projects\PhD\Drone_Air_Sim\universal-uav-rl
$env:PYTHONPATH="f:\Projects\PhD\Drone_Air_Sim\universal-uav-rl"
.\.venv\Scripts\python.exe training\train_rllib_ppo.py
```

### 6. Monitor Progress
```bash
# TensorBoard (in another terminal)
tensorboard --logdir models/logs

# View in browser
# http://localhost:6006
```

## Training Progress Indicators

### Early Training (Iterations 1-20)
- **Rewards**: Highly negative (-2000 to -500)
- **Behavior**: Drones hover near spawn (PID dominant), small random movements
- **Collisions**: Occasional due to exploration
- **Goal Achievement**: Rare

### Mid Training (Iterations 20-100)
- **Rewards**: Improving (-500 to -100)
- **Behavior**: Deliberate movement toward goals, swarm starting to coordinate
- **Collisions**: Decreasing as policy learns avoidance
- **Goal Achievement**: 10-30%

### Late Training (Iterations 100-500)
- **Rewards**: Positive trend (-100 to +50)
- **Behavior**: Efficient paths, swarm moves together, smooth trajectories
- **Collisions**: Rare (<5%)
- **Goal Achievement**: 50-80%

### Well-Trained (Iterations 500+)
- **Rewards**: Consistently positive (+50 to +150)
- **Behavior**: Coordinated swarm navigation, efficient paths, collision-free
- **Collisions**: Very rare (<1%)
- **Goal Achievement**: >80%

## Checkpoints & Logging

### Checkpoints
- Saved every 50 iterations to `models/checkpoints/run_YYYYMMDD_HHMMSS/`
- Contains:
  - Policy network weights
  - Value function weights
  - Optimizer state
  - Training statistics

### Episode Logs
- Detailed CSV logs in `data/episodes/run_YYYYMMDD_HHMMSS/`
- One file per episode: `episode_<uuid>.csv`
- Columns: timestep, agent positions/velocities, actions, rewards, collisions, etc.
- See [metrics.md](metrics.md) for full column list

### TensorBoard Logs
- Real-time metrics in `models/logs/`
- View with: `tensorboard --logdir models/logs`
- Includes:
  - Episode rewards (mean, min, max)
  - Success/collision rates
  - Swarm metrics (cohesion, alignment)
  - Policy/value losses
  - Learning rate

## Troubleshooting

### Drones Falling Immediately
- **Check**: `use_pid_assist: true` in `config/env.yaml`
- **Check**: `pid_blend_weight` ≥ 0.3 for sufficient stabilization
- **Check**: AirSim is running and connected

### Training Stalls / No Samples
- **Check**: `num_env_runners: 0` (not 1 or higher with single AirSim)
- **Check**: AirSim not frozen (check Unreal window)
- **Check**: No msgpack errors in logs

### High Collision Rate
- **Increase**: `d_safe` (e.g., 5.0 → 8.0)
- **Increase**: `beta` (near-miss penalty)
- **Increase**: `w_ttc` (predictive safety)
- **Reduce**: `num_agents` temporarily (e.g., 5 → 3)

### Drones Not Reaching Goals
- **Increase**: `alpha` (progress reward)
- **Reduce**: `w_cohesion` (less grouping, more exploration)
- **Increase**: `goal_radius` (easier success criterion)
- **Check**: Goals are reachable (not outside bounds)

### Jerky/Oscillating Behavior
- **Increase**: `action_frequency` (e.g., 10 → 15 or 20)
- **Increase**: `lambda` (smoothness penalty)
- **Increase**: `pid_blend_weight` (more PID damping)

### Slow Training
- **Disable rendering**: `"ViewMode": "NoDisplay"` in `airsim_settings.json`
- **Reduce**: `train_batch_size` (8000 → 4000)
- **Check**: GPU utilization (should be >80%)

## Advanced Features

### 2D Curriculum Learning
Start with planar motion (lock Z-axis) for faster initial learning:
```yaml
curriculum_2d_mode: true  # in config/env.yaml
```
After decent 2D performance, switch to full 3D:
```yaml
curriculum_2d_mode: false
```

### Adjusting PID Assistance
Reduce PID blend as policy improves:
```yaml
# Early training
pid_blend_weight: 0.4  # More stability

# Mid training  
pid_blend_weight: 0.3  # Default balance

# Late training
pid_blend_weight: 0.1  # Minimal assistance, RL dominant
```

### Custom Reward Tuning
Modify weights in `config/env.yaml` to emphasize different behaviors:
```yaml
# More aggressive goal-seeking
alpha: 30.0
w_cohesion: 0.05

# More conservative safety
beta: 1.0
w_ttc: 0.3
d_safe: 8.0

# Smoother control
lambda: 0.05
action_frequency: 15
```

## Evaluation

### Test Trained Policy
```bash
python evaluation/evaluate_policy_airsim.py \
  --checkpoint models/checkpoints/run_20260202_141049/checkpoint_000500 \
  --num_episodes 10
```

### Analyze Episode Data
```bash
# Summary statistics
python evaluation/analyze_episodes.py --run run_20260202_141049

# Movement patterns
python evaluation/analyze_movement.py --episode data/episodes/run_20260202_141049/episode_abc123.csv

# Reward breakdown
python evaluation/diagnose_rewards.py --run run_20260202_141049
```

### Visualize Trajectories
```bash
python evaluation/visualize_logs.py \
  --episode data/episodes/run_20260202_141049/episode_abc123.csv \
  --output trajectory.png
```

## Best Practices

1. **Start Small**: Train with 3 agents first, then scale to 5
2. **Monitor Early**: Check first 10 iterations - rewards should improve
3. **Use TensorBoard**: Real-time monitoring prevents wasted training time
4. **Save Often**: Checkpoints every 50 iterations is good balance
5. **Test Incrementally**: Evaluate every 100 iterations, adjust hyperparameters
6. **Document Changes**: Log what you change and why in training notes
7. **Use Smoke Tests**: Run `smoke_test_env.py` before long training runs

## Files Reference

### Core Training
- `training/train_rllib_ppo.py` - Main training script
- `training/callbacks.py` - Custom RLlib callbacks for metrics
- `config/training.yaml` - PPO hyperparameters
- `config/env.yaml` - Environment and reward configuration

### Environment
- `envs/universal_uav_env.py` - PettingZoo parallel environment
- `envs/airsim_client.py` - AirSim communication wrapper
- `envs/pid_controller.py` - PID stabilization controllers
- `envs/utils_observation.py` - Observation space construction
- `envs/utils_reward.py` - Reward function components
- `envs/utils_logging.py` - Episode CSV logging

### Evaluation
- `evaluation/analyze_episodes.py` - Batch episode analysis
- `evaluation/analyze_movement.py` - Trajectory analysis
- `evaluation/diagnose_rewards.py` - Reward component debugging
- `evaluation/evaluate_policy_airsim.py` - Policy testing in AirSim
- `evaluation/visualize_logs.py` - Trajectory visualization

### Testing
- `tests/test_frames.py` - Coordinate frame transformations
- `tests/test_obs_shapes.py` - Observation space validation
- `tests/test_reward_sanity.py` - Reward function unit tests
- `scripts/smoke_test_env.py` - Quick environment verification
- `test_pid.py` - PID controller verification
- `test_cumulative_actions.py` - Cumulative action demo
- `test_env_cumulative.py` - Full integration test
