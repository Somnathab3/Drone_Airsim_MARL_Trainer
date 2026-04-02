# Cooperative RL Extension for Multi-UAV CD&R

This extension upgrades the multi-UAV environment with explicit cooperative capabilities for online conflict detection and resolution (CD&R).

## 1. Cooperative Intent-Sharing Observation Layer
The observation space is expanded from `33 + 8*K` to `33 + 14*K`. Each of the $K$ nearest neighbors now provides 12 features:
- **Relative Position** (3): body-frame [x, y, z]
- **Relative Velocity** (3): body-frame [vx, vy, vz]
- **Neighbor Goal Direction** (3): body-frame [x, y, z] of the neighbor's current waypoint/goal.
- **Neighbor Waypoint Distance** (1): tanh-normalized distance to its current goal.
- **Neighbor Priority** (1): Local right-of-way score [0, 1].
- **Communication Freshness** (1): Age of the message [1.0 = fresh, 0.0 = stale].

## 2. Communication Model
A receiver-centric communication model simulates real-world networking constraints:
- **Packet Drops**: Messages from neighbors can be dropped with probability `drop_prob`.
- **Delay**: Messages are delayed by `delay_steps` (discrete simulation steps).
- **Radius**: Communication is limited to a configurable `radius` (meters).
- **Staleness**: If a message is dropped, the environment retains the last successful message up to `stale_limit` steps, but its `freshness` score decreases.

## 3. Local Arbitration & Action Biasing
The environment includes a rule-based arbitration layer to stabilize cooperative RL behavior:
- **Priority Scoring**: Calculated as `(100 - DistToGoal)/100 + Speed/10`. Drones closer to their goal or moving faster have higher priority.
- **Collision Anticipation**: Uses simple TTC (Time-To-Collision) to identify high-risk drone pairs.
- **Yielding Bias**: When a conflict is detected (TTC < threshold), the lower-priority drone (yielder) receives a velocity bias:
    - **Slow down**: Forward velocity is reduced.
    - **Steer away**: A lateral nudge is applied to increase separation.
- **Action Shield**: If `use_action_shield` is enabled, these corrections are more aggressive to prevent collisions.

## 4. Cooperative Reward Refinement
Rewards are now divided into three tiers:
### Ego Rewards
- Progress, goal reach, time penalty, action smoothness.
### Swarm Rewards
- Cohesion, velocity alignment, TTC-based predictive safety.
### Cooperative Rewards
- **Conflict Resolution Bonus** (`w_coop_resolve`): Reward for increasing separation when TTC is low.
- **Priority-aware Yielding** (`w_priority_yield`): Reward for lower-priority agents that successfully reduce speed/steer away in a conflict.
- **Team Progress** (`w_team_progress`): A global reward based on the average progress rate of all active agents, encouraging "socially aware" pathing.
- **Deadlock Penalty** (`w_deadlock_penalty`): Penalty for agents that stop moving when near neighbors.

## 5. New Configuration Flags (`config/env.yaml`)
```yaml
cooperation:
  use_intent_sharing: true      # Toggle intent features in obs
  use_priority_negotiation: true # Toggle local arbitration
  use_action_shield: false       # Toggle hard safety corrections
  shield_threshold_ttc: 3.0      # TTC threshold for arbitration
  
communication:
  drop_prob: 0.1                 # Packet loss probability
  delay_steps: 1                 # Message delay
  radius: 50.0                   # Max comm distance
```

## 6. Cooperative Metrics
Aggregated at the team level in RLlib:
- `team_success_rate`: Percentage of agents reaching goal.
- `conflict_resolution_count`: Number of successful deconfliction maneuvers.
- `team_progress_mean`: Average collective progress rate.
- `deadlock_rate`: Frequency of agents stalling near neighbors.
