# Reward Design

The reward function incentivizes reaching the goal while avoiding collisions, maintaining swarm cohesion, and optimizing for time/smoothness.

## Core Components

### 1. Progress Reward
Encourages moving closer to the goal at every step.
```
r_prog = alpha * (dist_prev - dist_curr)
```
* `alpha` (default 20.0) scales the meter-for-meter gain
* Increased from 10.0 for stronger goal-seeking signal

### 2. Goal Bonus
One-time reward upon entering `goal_radius`.
```
+R_goal = 100.0
```
* `goal_radius` (default 5.0m, increased for easier initial learning)

### 3. Collision Penalty
**Smart penalty system that only penalizes NEW collisions, not repeated ones.**

**Collision Detection:**
- Uses AirSim's `simGetCollisionInfo()` API
- Detects ALL collision types:
  - Other drones
  - Buildings and walls
  - Ground/floor
  - Ceiling (indoor environments)
  - Any static obstacles

**Penalty Logic:**
```
if is_new_collision:
    -R_collision = 50.0  (one-time penalty)
elif still_colliding:
    only -eta time penalty (no additional collision penalty)
```

**Why This Design?**
- **Problem**: Drones stuck on ground for 10-20 steps would accumulate -500 to -1000 penalty
- **Solution**: Only penalize the FIRST collision detection
- **Benefit**: Drone can recover from collision without catastrophic reward damage
- **Learning**: Policy learns to avoid collisions, not just to avoid being stuck

**Grace Period:**
- First **N steps** after spawn (N = `action_frequency`, default 10)
- Collisions ignored during grace period
- **Reason**: With cumulative actions, drone has no control until step N
  - Actions accumulate for 10 steps
  - First action sent at step 10
  - Penalizing collisions before control is unfair
- Allows PID controller to stabilize
- Configurable: Automatically matches `action_frequency` in `config/env.yaml`

**Collision State Tracking:**
```python
# Step 1: Collision detected → penalty = -50.0, state = "colliding"
# Step 2: Still colliding → penalty = -0.1 (time only), state = "stuck"
# Step 3: Collision cleared → state = "free"
# Step 4: New collision → penalty = -50.0 (new collision), state = "colliding"
```

### 4. Safe Separation Penalty (Near Miss)
Smooth penalty when closer than `d_safe` to any neighbor.
```
r_near = -beta * (d_safe - sep_min)^2  (if sep_min < d_safe)
```
* `beta` (default 0.5, reduced from 1.0 for gentler penalties)
* `d_safe` (default 5.0m, increased from 3.0m for more breathing room)

### 5. Time Penalty
Small constant penalty per step to encourage efficiency.
```
-eta = 0.1
```

## Swarm Coordination Components (Section 4.1)

### 6. Cohesion Reward
Encourages agents to stay near swarm centroid (flock together).
```
r_cohesion = -w_cohesion * max(0, dist_to_centroid - r_cohesion_max)^2
```
* `w_cohesion` (default 0.1)
* `r_cohesion_max` (default 20.0m) - maximum acceptable distance from centroid
* Quadratic penalty for straying too far from group

### 7. Velocity Alignment Reward
Rewards flying in same direction as swarm (coordinated movement).
```
r_velocity_align = w_velocity_align * cos(angle_between_velocities)
```
* `w_velocity_align` (default 0.05)
* Cosine similarity: +1 when aligned, -1 when opposite
* Encourages synchronized swarm motion

### 8. Time-to-Collision (TTC) Safety Reward
Predictive safety: penalizes trajectories leading to future collisions.
```
ttc = distance / relative_velocity
r_ttc = -w_ttc * exp(-ttc / ttc_tau)  (if ttc < ttc_tau)
```
* `w_ttc` (default 0.2)
* `ttc_tau` (default 2.0 seconds) - time horizon for collision prediction
* Exponential penalty for low time-to-collision
* Proactive collision avoidance (better than reactive near-miss penalty)

## Action Penalties

### 9. Action Magnitude Penalty (Energy Cost)
```
r_action_cost = -gamma * ||action||
```
* `gamma` (default 0.005, reduced from 0.01)
* Encourages energy-efficient control

### 10. Action Smoothness Penalty (Jerk Reduction)
```
r_smoothness = -lambda * ||action_t - action_{t-1}||
```
* `lambda` (default 0.01, reduced from 0.05)
* Encourages smooth control transitions
* Note: With cumulative actions (every 10 steps), smoothness is less critical

## Altitude Control (Gentle Guidance)

### 11. Altitude Preference Reward
```
if altitude < altitude_min: penalty = -omega * (altitude_min - altitude)^2
if altitude > altitude_max: penalty = -omega * (altitude - altitude_max)^2
if altitude in [preferred_min, preferred_max]: small_bonus = +omega * 0.1
```
* `omega` (default 0.3, gentle guidance)
* Soft boundaries: `altitude_min=1.0m`, `altitude_max=30.0m`
* Preferred band: `[3.0m, 25.0m]`

## Additional Features

### Heading Alignment Reward
```
r_heading = zeta * cos(angle_to_goal)
```
* `zeta` (default 1.5)
* Encourages facing toward goal (better than sideways flight)

### Catastrophic Failure Termination
Episode terminates if cumulative reward drops below threshold:
```
if cumulative_reward < -500.0: terminate_episode()
```
* Prevents wasting time on clearly failed episodes

## Terminal Conditions
* **Success**: Distance to goal < `goal_radius`.
* **Failure**: Collision detected (AirSim API).
* **Timeout**: Max steps reached.
