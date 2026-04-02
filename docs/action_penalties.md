# Action Control System

## Overview
The system uses a hybrid control approach combining RL policy learning with PID-assisted stabilization and cumulative action accumulation for smoother, more stable flight.

## Penalty Components

### 1. **Action Magnitude Penalty** (Energy Cost)
```
r_action_cost = -gamma * ||a||
```

- **Coefficient**: `gamma = 0.01` (in `config/env.yaml`)
- **Purpose**: Penalize large actions to discourage aggressive/wasteful control
- **Range**: Approximately [-0.2, 0] per step
  - Zero action: 0 penalty
  - Max action `||a|| = sqrt(10^2 + 5^2 + 5^2 + 2^2) = 12.45`: penalty = -0.12
- **Typical Impact**: -0.01 to -0.05 per step for moderate control

### 2. **Action Smoothness Penalty** (Jerk Reduction)
```
r_smoothness = -lambda * ||a_t - a_{t-1}||
```

- **Coefficient**: `lambda = 0.05` (in `config/env.yaml`)
- **Purpose**: Penalize rapid changes in actions to encourage smooth flight
- **Range**: Approximately [-0.6, 0] per step
  - No change: 0 penalty
  - Max change `||Δa|| = 2*12.45 = 24.9`: penalty = -1.25
- **Typical Impact**: -0.05 to -0.2 per step for smooth control

## Reward Component Summary

| Component | Coefficient | Typical Range | Purpose |
|-----------|-------------|---------------|---------|
| Progress | α = 10.0 | -5 to +5 | Reward getting closer to goal |
| Time penalty | η = 0.1 | -0.1 | Discourage hovering |
| **Action cost** | **γ = 0.01** | **-0.05 to 0** | **Energy efficiency** |
| **Smoothness** | **λ = 0.05** | **-0.2 to 0** | **Smooth control** |
| Near-miss | β = 1.0 | -25 to 0 | Avoid other drones |
| Collision | R_coll = 100 | -100 | Hard penalty |
| Goal | R_goal = 100 | +100 | Success bonus |

## Expected Impact

### Before Action Penalties
- Agents may use aggressive, jerky control
- Random oscillations due to exploration
- Wasted energy on unnecessary movements

### After Action Penalties
- Smoother trajectories with gradual velocity changes
- More energy-efficient paths
- Reduced control oscillations
- Better sample efficiency (smoother policies are easier to learn)

## Tuning Guidelines

### If drones are too conservative/slow:
- Reduce `gamma` (e.g., 0.005) or `lambda` (e.g., 0.02)

### If drones are still too jerky:
- Increase `lambda` (e.g., 0.1) for more smoothness emphasis

### If drones hover too much:
- Action penalties are working! Time penalty (`eta`) will push them forward

## Cumulative Action System

### Design Rationale
Instead of sending RL actions directly to AirSim every step, actions **accumulate** over multiple steps before being sent. This provides:

1. **Smoother Control**: Filters out high-frequency oscillations from exploration
2. **Reduced Command Frequency**: Less stress on AirSim communication
3. **Natural Integration**: Actions sum up, allowing gradual velocity changes
4. **Better Learning**: Policy learns to plan over horizons rather than single steps

### How It Works

**Action Frequency**: Every `N` steps (default: 10)

```
Steps 1-9:   Accumulate RL outputs (sum them up)
Step 10:     Send cumulative sum to AirSim, reset buffer
Steps 11-19: Accumulate again (fresh buffer)
Step 20:     Send next cumulative sum
```

**Example Timeline**:
```
Step 1:  RL outputs [2.0, 0.5, 1.0, 0.1]  → Buffer: [2.0, 0.5, 1.0, 0.1]
Step 2:  RL outputs [3.0, 0.5, 0.5, 0.2]  → Buffer: [5.0, 1.0, 1.5, 0.3]
Step 3:  RL outputs [1.0, -0.3, 0.8, 0.1] → Buffer: [6.0, 0.7, 2.3, 0.4]
...
Step 10: RL outputs [2.1, 0.4, 0.7, 0.12] → Buffer: [20.0, 3.2, 7.1, 1.27]
         ✓ SEND [20.0, 3.2, 7.1, 1.27] to AirSim!
         Reset buffer to [0, 0, 0, 0]
Step 11: RL outputs [1.5, 0.2, 0.9, 0.11] → Buffer: [1.5, 0.2, 0.9, 0.11]
         (AirSim still executing [20.0, 3.2, 7.1, 1.27])
```

**Configuration** (`config/env.yaml`):
```yaml
action_frequency: 10  # Send every 10 steps (default)
action_frequency: 5   # More frequent (faster response)
action_frequency: 20  # Less frequent (smoother, slower)
```

## PID-Assisted Hover Stabilization

### Purpose
Prevents drones from falling during early training when RL policy is random. Provides baseline stability while RL learns goal-seeking behavior.

### PID Controller Design

Three independent PID controllers per drone:

1. **Position Control (X, Y, Z)**:
   - P gain: 0.5-0.8 (proportional to position error)
   - I gain: 0.1-0.15 (eliminates steady-state error)
   - D gain: 0.2-0.3 (dampens oscillations)
   - Output: Velocity commands (m/s)

2. **Yaw Control**:
   - P gain: 1.0 (stronger for quick orientation)
   - I gain: 0.05 (gentle integral)
   - D gain: 0.1 (damping)
   - Output: Yaw rate (rad/s)

### Blending Strategy

```python
final_action = (1 - blend_weight) * rl_action + blend_weight * pid_action
```

- `blend_weight` (default: 0.3)
  - 30% PID stabilization
  - 70% RL policy exploration/exploitation

**Early Training** (random RL policy):
- PID dominates → drone hovers stably at spawn position
- RL explores small perturbations around hover

**Late Training** (learned RL policy):
- RL dominates → sophisticated goal-seeking and swarm coordination
- PID provides gentle stability corrections

**Configuration** (`config/env.yaml`):
```yaml
use_pid_assist: true        # Enable PID blending
pid_blend_weight: 0.3       # 30% PID, 70% RL
pid_blend_weight: 0.1       # Less assistance (advanced training)
pid_blend_weight: 0.5       # More assistance (early training)
```

## Implementation Details

**Files Modified**:
- `config/env.yaml`: Added `gamma`, `lambda`, `action_frequency`, `use_pid_assist`, `pid_blend_weight`
- `envs/pid_controller.py`: **NEW** - PID controller implementation
- `envs/utils_reward.py`: Added action cost and smoothness calculations
- `envs/universal_uav_env.py`: 
  - Cumulative action accumulation and periodic sending
  - PID controller instantiation and blending
  - Pass current and previous actions to reward function

**Data Flow**:
1. RL policy outputs action every step
2. PID computes hover stabilization command
3. Blend: `final = 0.7*RL + 0.3*PID`
4. Accumulate blended action in buffer
5. Every 10 steps: send accumulated sum to AirSim, reset buffer
6. Between sends: AirSim continues executing last command
7. Penalties computed and logged

**First Step Handling**:
- Previous action defaults to `None` on first step
- Smoothness penalty only applied when both actions available
- Magnitude penalty always applied (encourages gentle start)
- Cumulative buffer initialized to zeros
- Last sent action initialized to zeros (hover)

## Testing & Verification

**Test PID Controller**:
```bash
python test_pid.py
# Verifies PID outputs correct stabilization commands
```

**Test Cumulative Actions**:
```bash
python test_cumulative_actions.py
# Demonstrates action accumulation over 10 steps

python test_env_cumulative.py  
# Tests integration with environment
```
