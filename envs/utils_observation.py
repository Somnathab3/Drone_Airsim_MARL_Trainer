import numpy as np
from .velocity_obstacles import compute_velocity_obstacle_for_neighbors


def pad_or_trunc(x, n, pad_value=0.0, dtype=np.float32):
    """
    Pad or truncate array to exactly n elements.
    
    Args:
        x: Input array (can be None, empty, or any length)
        n: Target length
        pad_value: Value to use for padding
        dtype: Output dtype
    
    Returns:
        Array of exactly length n
    """
    if x is None:
        return np.full(n, pad_value, dtype=dtype)
    a = np.asarray(x, dtype=dtype).reshape(-1)
    if a.size == 0:
        return np.full(n, pad_value, dtype=dtype)
    if a.size < n:
        return np.pad(a, (0, n - a.size), constant_values=pad_value).astype(dtype)
    return a[:n]

def build_observation(agent_state, neighbors, goal_pos, config, prev_action=None):
    """
    Construct observation vector for a single agent.
    Section 4.2: Uses egocentric body-frame coordinates for neighbors.
    
    Args:
        agent_state: dict {pos, vel, yaw, centroid_pos, centroid_vel, ttc_min} in ENU
        neighbors: list of dicts {pos, vel} for K nearest neighbors
        goal_pos: [x, y, z] in ENU
        config: dict with parameters (d_ref, v_ref, K, R_min, R_gain, etc.)
        prev_action: previous action [vx, vy, vz, yaw_rate] for stability
    
    Returns:
        obs: np.ndarray
    """
    p_drone = np.array(agent_state['pos'])
    v_drone = np.array(agent_state['vel'])
    yaw = agent_state['yaw']
    p_goal = np.array(goal_pos)
    
    # --- Rotation helpers for egocentric (body frame) ---
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    
    def world_to_body(vec_world):
        """Rotate 3D vector from ENU world to body frame"""
        vx = vec_world[0] * cos_yaw - vec_world[1] * sin_yaw
        vy = vec_world[0] * sin_yaw + vec_world[1] * cos_yaw
        vz = vec_world[2]  # Z unchanged (yaw-only rotation)
        return np.array([vx, vy, vz])
    
    # --- Goal Features (Body Frame) ---
    goal_rel_world = p_goal - p_drone
    goal_rel_body = world_to_body(goal_rel_world)
    
    d = np.linalg.norm(goal_rel_world)
    d_ref = config.get('d_ref', 100.0)
    d_goal_close = config.get('d_goal_close', 5.0) # Feature for precision
    
    wp_dist_norm = np.tanh(d / d_ref)
    wp_dist_close_norm = np.tanh(d / d_goal_close) # Sensitive at close range
    
    # Normalized goal direction in body frame
    goal_dir_body_norm = goal_rel_body / (d + 1e-6)
    
    # --- Ownship Features (Body Frame) ---
    v_ref = config.get('v_ref', 10.0)
    v_body = world_to_body(v_drone)
    v_body_norm = np.tanh(v_body / v_ref)
    airspeed = np.linalg.norm(v_drone)
    airspeed_norm = np.tanh(airspeed / v_ref)
    
    # --- Derivative Features (Placeholder for now, usually needs history buffer) ---
    # Passed in agent_state should ideally include derivative info if computed outside.
    # For now, let's assume 'progress_rate' and 'safety_rate' are passed in or zero.
    progress_rate = agent_state.get('progress_rate', 0.0)
    safety_rate = agent_state.get('safety_rate', 0.0)
    ttc_min = agent_state.get('ttc_min', 10.0)  # Default 10s if not set
    # CRITICAL FIX: Clip inf to prevent NaN in tanh
    ttc_min_clipped = np.clip(ttc_min, -100.0, 100.0)
    ttc_min_norm = np.tanh(ttc_min_clipped / 5.0)  # Normalize around 5s
    
    # --- Centroid Features (Section 4.1, 4.2) ---
    centroid_pos = agent_state.get('centroid_pos', p_drone)
    centroid_vel = agent_state.get('centroid_vel', v_drone)
    
    centroid_rel_world = centroid_pos - p_drone
    centroid_rel_body = world_to_body(centroid_rel_world)
    centroid_rel_body_norm = np.tanh(centroid_rel_body / config.get('p_ref', 100.0))
    
    centroid_vel_body = world_to_body(centroid_vel)
    centroid_vel_body_norm = np.tanh(centroid_vel_body / v_ref)
    
    # --- Previous Action (Section 4.2) ---
    # Always pad to 4 dimensions for observation consistency
    if prev_action is not None:
        if len(prev_action) == 3:
            # Simplified action space: [speed, yaw_rate, climb_rate] - pad with zero for consistency
            prev_action_4d = np.concatenate([prev_action, [0.0]])
            prev_action_norm = prev_action_4d / np.array([10.0, 2.0, 2.0, 1.0])
        else:
            # Original action space: [vx, vy, vz, yaw_rate]
            prev_action_norm = np.array(prev_action) / np.array([10.0, 5.0, 5.0, 2.0])
    else:
        prev_action_norm = np.zeros(4)
    
    # --- Neighbor Features ---
    K = config.get('K', 5)
    obs_neighbors = []
    
    # Section 4.2: Neighbors in egocentric body frame
    # We need relative pos/vel and INTENT in body frame
    p_ref = config.get('p_ref', 100.0)
    v_ref_rel = config.get('v_ref_rel', 20.0)
    d_ref = config.get('d_ref', 100.0)
    
    # We assume 'neighbors' list contains pre-filtered K nearest
    for neighbor in neighbors[:K]:
        p_n = np.array(neighbor['pos'])
        v_n = np.array(neighbor['vel'])
        
        # Relative position and velocity in world frame
        rel_p_world = p_n - p_drone
        rel_v_world = v_n - v_drone
        
        # Transform to body frame (Section 4.2: egocentric)
        rel_p_body = world_to_body(rel_p_world)
        rel_v_body = world_to_body(rel_v_world)
        
        # Tanh scaling
        rel_p_norm = np.tanh(rel_p_body / p_ref)
        rel_v_norm = np.tanh(rel_v_body / v_ref_rel)
        
        obs_neighbors.extend(rel_p_norm)
        obs_neighbors.extend(rel_v_norm)

        # --- COOPERATIVE INTENT FEATURES ---
        # 1. Neighbor Goal Direction (in ownship body frame)
        n_goal_dir_world = neighbor.get('goal_dir', np.zeros(3))
        n_goal_dir_body = world_to_body(n_goal_dir_world)
        
        # 2. Neighbor Waypoint Distance
        n_wp_dist = neighbor.get('wp_dist', d_ref)
        n_wp_dist_norm = np.tanh(n_wp_dist / d_ref)
        
        # 3. Neighbor Priority (Right-of-way)
        n_priority = neighbor.get('priority', 0.0)
        
        # 4. Communication Freshness (1.0 = newest, 0.0 = stale/dead)
        n_freshness = neighbor.get('freshness', 1.0)
        
        obs_neighbors.extend(n_goal_dir_body)
        obs_neighbors.extend([n_wp_dist_norm, n_priority, n_freshness])
    
    # CRITICAL: Zero padding to ALWAYS have exactly K neighbors (fixed observation size)
    # Each neighbor = 12 features (3 pos + 3 vel + 3 goal_dir + 1 dist + 1 priority + 1 freshness)
    num_present = len(neighbors)
    if num_present < K:
        pad_size = (K - num_present) * 12
        obs_neighbors.extend([0.0] * pad_size)
    
    # Sanity check: ensure we have exactly K*12 neighbor features
    assert len(obs_neighbors) == K * 12, f"Neighbor features must be exactly {K*12}, got {len(obs_neighbors)}"
    
    # --- Obstacle Ray Features (8-directional) ---
    # CRITICAL: Use pad_or_trunc to ensure exactly 8 rays (robust against missing data)
    obstacle_rays_raw = agent_state.get('obstacle_rays', np.full(8, 50.0))
    obstacle_rays = pad_or_trunc(obstacle_rays_raw, 8, pad_value=50.0)
    obstacle_rays_norm = np.tanh(obstacle_rays / 25.0)  # Normalize around 25m
    
    # --- Velocity Obstacle Features ---
    # CRITICAL: Use pad_or_trunc to ensure exactly K elements (fixes 62 vs 72 dimension bug)
    vo_collision_flags_raw = agent_state.get('vo_collision_flags', np.zeros(K, dtype=np.float32))
    vo_collision_flags = pad_or_trunc(vo_collision_flags_raw, K, pad_value=0.0) # Pad with 0.0 (no collision)
    
    vo_ttcs_raw = agent_state.get('vo_ttcs', np.full(K, 10.0, dtype=np.float32))
    vo_ttcs = pad_or_trunc(vo_ttcs_raw, K, pad_value=1.0)  # Pad with 1.0 (safe/max normalized TTC)
    # Note: If passing raw values, pad with large number (e.g. 100.0)
    # But here we assume we might get raw or normalized. 
    # Let's handle raw mapping first:
    
    # Clip inf to large finite value BEFORE tanh to prevent NaN!
    vo_ttcs_clipped = np.clip(vo_ttcs, -100.0, 100.0)
    
    # Re-doing robustly:
    vo_ttcs_raw = agent_state.get('vo_ttcs', np.array([]))
    vo_ttcs_padded = pad_or_trunc(vo_ttcs_raw, K, pad_value=100.0) # 100s = safe
    vo_ttcs_clipped = np.clip(vo_ttcs_padded, -100.0, 100.0)
    vo_ttcs_norm = np.tanh(vo_ttcs_clipped / 5.0) 
    
    
    # --- Waypoint Features (PHASE 1) ---
    waypoint_dir_raw = agent_state.get('waypoint_direction', goal_dir_body_norm[:3])
    waypoint_dist_raw = agent_state.get('waypoint_distance', d)
    
    # Ensure valid waypoint direction (handle zero-length case)
    waypoint_dir_norm_magnitude = np.linalg.norm(waypoint_dir_raw)
    if waypoint_dir_norm_magnitude > 1e-6:
        waypoint_dir = waypoint_dir_raw / waypoint_dir_norm_magnitude
    else:
        waypoint_dir = np.array([1.0, 0.0, 0.0])  # Default: forward
    
    waypoint_dist_norm = np.tanh(waypoint_dist_raw / d_ref)
    
    # Assemble observation vector (Section 4.2: comprehensive egocentric obs)
    obs = [
        # Goal features (5) - Increased from 4
        wp_dist_norm,
        wp_dist_close_norm, # NEW: Precision feature
        *goal_dir_body_norm,  # 3 values: normalized goal direction in body frame
        
        # Waypoint features (4) - PHASE 1: RRT* guidance
        waypoint_dist_norm,
        *waypoint_dir,  # 3 values: normalized waypoint direction
        
        # Ownship velocity in body frame (3)
        *v_body_norm,
        
        # Derivative features (3)
        progress_rate, 
        safety_rate,
        ttc_min_norm,
        
        # Centroid features (6)
        *centroid_rel_body_norm,  # 3 values: centroid position in body frame
        *centroid_vel_body_norm,  # 3 values: centroid velocity in body frame
        
        # Previous action (4)
        *prev_action_norm,
        
        # Obstacle rays (8)
        *obstacle_rays_norm,
        
        # VO features (K collision flags + K TTCs)
        *vo_collision_flags,  # Already padded to K
        *vo_ttcs_norm,        # Already padded to K
    ] + obs_neighbors  # K * 12 neighbor features
    
    obs_array = np.array(obs, dtype=np.float32)
    
    # CRITICAL: Guarantee fixed observation size
    # 33 + 14 * K (where K=5) -> 33 + 70 = 103
    expected_size = 33 + 14 * K
    
    # Safe padding/truncation to ensure EXACTLY expected_size
    if len(obs_array) != expected_size:
        # Log mismatch but fix it silently to prevent crash
        if len(obs_array) < expected_size:
            obs_array = np.pad(obs_array, (0, expected_size - len(obs_array)), constant_values=0.0)
        else:
            obs_array = obs_array[:expected_size]
            
    # CRITICAL: Safety check for NaN/inf values
    if not np.all(np.isfinite(obs_array)):
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return obs_array

def get_observation_space_shape(config):
    K = config.get('K', 5)
    # 5 (goal) + 4 (waypoint) + 3 (vel) + 3 (derivatives) + 6 (centroid) + 4 (prev_action) + 8 (rays) + K (VO flags) + K (TTCs) + K*12 (neighbors)
    # = 33 + 14 * K
    return (33 + 14 * K,)
