import numpy as np

def compute_reward(agent_state, goal_pos, prev_state, config, info):
    """
    Compute reward for single agent.
    
    Args:
        agent_state: dict {pos, vel, yaw, collision}
        goal_pos: [x,y,z] ENU
        prev_state: dict {dist_to_goal, prev_action} (from previous step)
        config: reward coefficients
        info: dict to update with metrics (should contain 'action', 'prev_action')
        
    Returns:
        reward: float
        terminated: bool
        truncated: bool
    """
    # Config
    R_goal = config.get('R_goal', 100.0)
    R_collision = config.get('R_collision', 100.0)
    # Weights
    alpha = config.get('alpha', 1.0)     # Progress
    beta = config.get('beta', 0.5)       # Safety/Near-miss
    eta = config.get('eta', 0.1)         # Time penalty
    gamma = config.get('gamma', 0.01)    # Action magnitude penalty
    lambda_smooth = config.get('lambda', 0.05)  # Action smoothness penalty
    zeta = config.get('zeta', 0.3)       # Heading alignment reward
    omega = config.get('omega', 2.0)     # Altitude control reward
    
    d_safe = config.get('d_safe', 2.0)
    goal_radius = config.get('goal_radius', 2.0)
    catastrophic_threshold = config.get('catastrophic_reward_threshold', -3000.0)
    
    # Altitude constraints
    altitude_min = config.get('altitude_min', 3.0)
    altitude_max = config.get('altitude_max', 25.0)
    altitude_preferred_min = config.get('altitude_preferred_min', 5.0)
    altitude_preferred_max = config.get('altitude_preferred_max', 20.0)
    
    # State
    p = np.array(agent_state['pos'])
    v = np.array(agent_state['vel'])
    g = np.array(goal_pos)
    dist = np.linalg.norm(p - g)
    
    # --- Terminal Conditions ---
    collision = agent_state['collision']
    reached_goal = dist < goal_radius
    
    terminated = False
    reward = 0.0
    
    # --- Reward Components ---
    
    # 1. Progress Reward
    d_prev = prev_state.get('dist_to_goal', dist)
    
    # Safety guard: ensure finite values
    if not np.isfinite(d_prev):
        d_prev = dist
    if not np.isfinite(dist):
        dist = 0.0
    
    r_prog = alpha * (d_prev - dist)
    reward += r_prog
    
    # 1.5. Velocity towards goal bonus (helps early exploration)
    # Reward for moving in the right direction
    if dist > goal_radius:
        goal_direction_normalized = (g - p) / (dist + 1e-6)
        velocity_towards_goal = np.dot(v, goal_direction_normalized)
        
        # Reward positive velocity towards goal
        if velocity_towards_goal > 0:
            r_velocity_bonus = 0.1 * velocity_towards_goal  # Small bonus for moving towards goal
            reward += r_velocity_bonus
            info['r_velocity_bonus'] = r_velocity_bonus
    
    # 2. Time Penalty
    reward -= eta
    
    # 3. Collision / Goal
    # Check if this is a NEW collision (not repeated penalty for being stuck)
    is_new_collision = collision and not prev_state.get('was_colliding', False)
    
    if is_new_collision:
        reward -= R_collision
        info['outcome'] = 'collision'
        info['is_new_collision'] = True
        terminated = True  # CRITICAL FIX: Terminate immediately on collision
    elif collision:
        # Still in collision from previous step - should have terminated already
        # But if for some reason it didn't (e.g. multi-agent processing order), terminate now
        reward -= R_collision * 0.1 # Small penalty for sticking
        info['outcome'] = 'stuck_collision'
        info['is_new_collision'] = False
        terminated = True  # Ensure termination
    elif reached_goal:
        reward += R_goal
        terminated = True
        info['outcome'] = 'success'
    
    # 4. Near-miss / Separation penalty with improved shaping
    # Progressive penalty that gets stronger as drones get closer
    sep_min = info.get('sep_min', float('inf'))
    if sep_min < d_safe:
        # Quadratic penalty - gets much worse as separation decreases
        # At d_safe: penalty = 0
        # At d_safe/2: penalty = -beta * (d_safe/2)^2
        # At 0: penalty = -beta * d_safe^2
        separation_violation = d_safe - sep_min
        r_near = -beta * separation_violation**2
        reward += r_near
        info['r_near'] = r_near
        
        # Add warning levels for debugging
        if sep_min < d_safe * 0.3:
            info['collision_warning'] = 'critical'
        elif sep_min < d_safe * 0.5:
            info['collision_warning'] = 'high'
        else:
            info['collision_warning'] = 'moderate'
    
    # 4.5. Heading alignment reward (CRITICAL for goal-seeking)
    # Reward for aligning heading with goal direction
    goal_direction = g - p
    horizontal_dist = np.linalg.norm(goal_direction[:2])
    
    if horizontal_dist > 1.0:  # Only apply if goal is >1m away horizontally
        # Goal bearing in ENU (angle from +X axis)
        goal_bearing_enu = np.arctan2(goal_direction[1], goal_direction[0])
        
        # Current heading (yaw in ENU)
        current_heading = agent_state['yaw']
        
        # Heading error (wrapped to [-pi, pi])
        heading_error = np.arctan2(np.sin(goal_bearing_enu - current_heading),
                                   np.cos(goal_bearing_enu - current_heading))
        
        # Reward for small heading error
        # When aligned (error=0): reward = +zeta
        # When perpendicular (error=+/- pi/2): reward = +zeta/2
        # When opposite (error=+/- pi): reward = 0
        r_heading = zeta * (np.pi - abs(heading_error)) / np.pi
        reward += r_heading
        info['r_heading'] = r_heading
        info['heading_error_deg'] = np.degrees(abs(heading_error))
    
    # 4.6. Altitude control reward (GENTLE GUIDE - not harsh constraint)
    # Encourage staying in safe altitude band without overly penalizing exploration
    current_altitude = p[2]  # ENU z-up
    
    if current_altitude < altitude_min:
        # Getting dangerously low - gentle warning
        r_altitude = -omega * (altitude_min - current_altitude)
        info['altitude_warning'] = 'too_low'
    elif current_altitude > altitude_max:
        # Getting dangerously high - gentle warning
        r_altitude = -omega * (current_altitude - altitude_max)
        info['altitude_warning'] = 'too_high'
    elif altitude_preferred_min <= current_altitude <= altitude_preferred_max:
        # In preferred band - tiny positive reward (encouragement only)
        r_altitude = omega * 0.05
        info['altitude_warning'] = 'optimal'
    else:
        # Outside preferred but still safe - very small nudge back (don't interfere with learning)
        r_altitude = 0.0  # No penalty - let other rewards guide
        info['altitude_warning'] = 'acceptable'
    
    reward += r_altitude
    info['r_altitude'] = r_altitude
    info['altitude'] = current_altitude
    
    # === SWARM COORDINATION REWARDS (Section 4.1) ===
    
    # 7. Cohesion to centroid (prevents dispersion)
    centroid_pos = agent_state.get('centroid_pos', None)
    if centroid_pos is not None:
        w_cohesion = config.get('w_cohesion', 0.1)
        r_cohesion_max = config.get('r_cohesion_max', 20.0)
        
        dist_to_centroid = np.linalg.norm(p - centroid_pos)
        cohesion_violation = max(0, dist_to_centroid - r_cohesion_max)
        
        # Linear-quadratic transition to prevent explosion
        # If violation > 10m, switch to linear
        if cohesion_violation > 10.0:
            # Value at 10m: 10^2 = 100
            # Slope at 10m: 2*10 = 20
            # Linear: 100 + 20 * (violation - 10)
            penalty_val = 100.0 + 20.0 * (cohesion_violation - 10.0)
        else:
            penalty_val = cohesion_violation**2
            
        r_cohesion = -w_cohesion * penalty_val
        
        # Hard clamp to prevent catastrophic episodes during early random exploration
        r_cohesion = max(r_cohesion, -200.0)
        
        reward += r_cohesion
        info['r_cohesion'] = r_cohesion
        info['dist_to_centroid'] = dist_to_centroid
    
    # 8. Velocity alignment (synchronizes motion)
    centroid_vel = agent_state.get('centroid_vel', None)
    if centroid_vel is not None and np.linalg.norm(centroid_vel) > 0.1:
        w_velocity_align = config.get('w_velocity_align', 0.05)
        
        # Cosine similarity between agent velocity and swarm average
        v_norm = np.linalg.norm(v)
        cv_norm = np.linalg.norm(centroid_vel)
        
        if v_norm > 0.1:  # Only apply when moving
            cos_sim = np.dot(v, centroid_vel) / (v_norm * cv_norm + 1e-6)
            r_velocity_align = w_velocity_align * cos_sim
            reward += r_velocity_align
            info['r_velocity_align'] = r_velocity_align
            info['velocity_alignment'] = cos_sim
    
    # 9. Predictive TTC penalty (prevents late panic avoidance)
    ttc_min = agent_state.get('ttc_min', float('inf'))
    if ttc_min < float('inf'):
        w_ttc = config.get('w_ttc', 0.2)
        ttc_tau = config.get('ttc_tau', 2.0)
        
        # Exponential penalty that gets stronger as TTC decreases
        r_ttc = -w_ttc * np.exp(-ttc_min / ttc_tau)
        reward += r_ttc
        info['r_ttc'] = r_ttc
        info['ttc_min'] = ttc_min
    
    # === END SWARM REWARDS ===
    
    # PHASE 0: VO-based collision risk (gradient-based, not binary)
    vo_ttcs = agent_state.get('vo_ttcs', np.array([]))
    vo_cpa_dists = agent_state.get('vo_cpa_dists', np.array([]))
    
    if len(vo_ttcs) > 0:
        from .velocity_obstacles import compute_collision_risk_reward
        
        w_collision_risk = config.get('w_collision_risk', 0.5)
        tau_collision = config.get('tau_collision', 2.0)
        
        r_vo_risk = compute_collision_risk_reward(
            vo_ttcs, vo_cpa_dists, tau=tau_collision, weight=w_collision_risk
        )
        reward += r_vo_risk
        info['r_vo_risk'] = r_vo_risk
        info['vo_min_ttc'] = float(np.min(vo_ttcs)) if len(vo_ttcs) > 0 else 10.0
    
    # PHASE 0: Obstacle proximity shaping (gradient-based)
    obstacle_rays = agent_state.get('obstacle_rays', np.full(8, 50.0))
    min_obs_dist = np.min(obstacle_rays)
    d_obs_safe = config.get('d_obs_safe', 3.0)
    w_obs = config.get('w_obs', 0.5)
    
    if min_obs_dist < d_obs_safe:
        # Exponential penalty that gets stronger as obstacle gets closer
        r_obs_proximity = -w_obs * np.exp(-min_obs_dist / 2.0)
        reward += r_obs_proximity
        info['r_obs_proximity'] = r_obs_proximity
    
    info['min_obs_dist'] = min_obs_dist
    
    # PHASE 1: Path efficiency reward (penalize deviation from planned path)
    waypoint_dir = agent_state.get('waypoint_direction', None)
    if waypoint_dir is not None and np.linalg.norm(v) > 0.5:
        w_path_efficiency = config.get('w_path_efficiency', 0.3)
        
        # Compute alignment between velocity and waypoint direction
        v_norm = v / np.linalg.norm(v)
        alignment = np.dot(v_norm, waypoint_dir)  # Cosine similarity
        
        # Reward alignment (1.0 = perfect, -1.0 = opposite)
        r_path_efficiency = w_path_efficiency * (alignment - 0.5)  # Centered around 0
        reward += r_path_efficiency
        info['r_path_efficiency'] = r_path_efficiency
        info['path_alignment'] = alignment
    
    # 5. Action magnitude penalty (energy cost)
    # Penalize large actions to encourage efficient control
    current_action = info.get('action', None)
    if current_action is not None:
        action_magnitude = np.linalg.norm(current_action)
        r_action_cost = -gamma * action_magnitude
        reward += r_action_cost
        info['r_action_cost'] = r_action_cost
    
    # 6. Action smoothness penalty (jerk reduction)
    # Penalize rapid changes in actions to encourage smooth control
    prev_action = prev_state.get('prev_action', None)
    if current_action is not None and prev_action is not None:
        action_change = np.linalg.norm(current_action - prev_action)
        r_smoothness = -lambda_smooth * action_change
        reward += r_smoothness
        info['r_smoothness'] = r_smoothness
        
    # Logging info
    info['reward'] = reward
    info['r_prog'] = r_prog
    info['dist_to_goal'] = dist
    
    # Check cumulative reward for catastrophic failure
    cumulative_reward = info.get('cumulative_reward', 0.0)
    if cumulative_reward < catastrophic_threshold:
        terminated = True
        info['outcome'] = 'catastrophic_failure'
    
    # --- COOPERATIVE REWARDS (CD&R Extension) ---
    w_coop_resolve = config.get('w_coop_resolve', 1.0)
    w_priority_yield = config.get('w_priority_yield', 0.5)
    w_team_progress = config.get('w_team_progress', 0.2)
    w_deadlock = config.get('w_deadlock_penalty', 0.5)

    # 10. Team Progress (Cooperative Goal seeking)
    # Reward based on average progress of all agents
    # This is passed via info or agent_state if pre-calculated
    team_progress = agent_state.get('team_avg_progress', 0.0)
    r_team = w_team_progress * team_progress
    reward += r_team
    info['r_team_progress'] = r_team

    # 11. Conflict Resolution Bonus
    # If agent was in a high-risk situation (low TTC) and separation is now increasing
    ttc_min = agent_state.get('ttc_min', float('inf'))
    safety_rate = agent_state.get('safety_rate', 0.0) # current_sep - prev_sep
    
    if ttc_min < 5.0 and safety_rate > 0:
        # Separation is increasing while in conflict zone!
        r_coop_resolve = w_coop_resolve * safety_rate
        reward += r_coop_resolve
        info['r_coop_resolve'] = r_coop_resolve
        info['resolution_active'] = True
    
    # 12. Priority-aware Yielding Reward
    # If arbitration layer is active and this agent is "yielding"
    is_yielding = agent_state.get('is_yielding', False)
    is_maintainer = agent_state.get('is_maintainer', False)
    
    if is_yielding:
        # Reward for actually slowing down or steering away when asked to yield
        # Check longitudinal speed reduction
        v_body = agent_state.get('vel_body', [0,0,0])[0]
        if v_body < 2.0: # Arbitrary threshold for "yielding speed"
            r_yield = w_priority_yield * 0.5
            reward += r_yield
            info['r_priority_yield'] = r_yield
    
    # 13. Deadlock Penalty
    # If stuck near another agent but not progressing
    if info.get('sep_min', 10.0) < d_safe * 2.0 and abs(r_prog) < 0.01:
        r_deadlock = -w_deadlock
        reward += r_deadlock
        info['r_deadlock_penalty'] = r_deadlock

    # Final safety guard: catch any numeric errors
    if not np.isfinite(reward):
        reward = -R_collision
        terminated = True
        info['outcome'] = 'numeric_error'
    
    return reward, terminated, False

