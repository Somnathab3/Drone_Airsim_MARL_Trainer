"""
Velocity Obstacles (VO) and collision prediction utilities
Based on RVO/ORCA algorithms for multi-agent collision avoidance
"""
import numpy as np
from typing import Tuple, Optional


def compute_velocity_obstacle(
    pos_self: np.ndarray,
    vel_self: np.ndarray,
    pos_other: np.ndarray,
    vel_other: np.ndarray,
    radius: float = 2.0
) -> Tuple[bool, float, float]:
    """
    Compute velocity obstacle features for collision avoidance.
    
    Args:
        pos_self: Position of self agent [x, y, z]
        vel_self: Velocity of self agent [vx, vy, vz]
        pos_other: Position of other agent [x, y, z]
        vel_other: Velocity of other agent [vx, vy, vz]
        radius: Combined collision radius (meters)
    
    Returns:
        is_in_collision_cone: True if vel_self is in VO cone
        ttc: Time-to-collision (inf if diverging)
        cpa_dist: Distance at closest point of approach
    """
    rel_pos = pos_other - pos_self
    rel_vel = vel_self - vel_other
    
    dist = np.linalg.norm(rel_pos)
    
    # Edge case: already colliding
    if dist < radius:
        return True, 0.0, 0.0
    
    # Compute time-to-collision (TTC)
    # TTC = time when distance is minimum (closing speed analysis)
    closing_speed = -np.dot(rel_pos, rel_vel) / (dist + 1e-9)
    
    if closing_speed <= 0:
        # Diverging or parallel motion
        ttc = float('inf')
        cpa_dist = dist
        is_in_cone = False
    else:
        # Approaching - compute TTC
        ttc = dist / closing_speed
        
        # Compute closest point of approach (CPA)
        # P_cpa = P_self + V_self * ttc
        # Distance at CPA
        pos_cpa_self = pos_self + vel_self * ttc
        pos_cpa_other = pos_other + vel_other * ttc
        cpa_dist = np.linalg.norm(pos_cpa_other - pos_cpa_self)
        
        # Check if collision will occur (CPA < radius)
        is_in_cone = cpa_dist < radius
    
    return is_in_cone, ttc, cpa_dist


def compute_velocity_obstacle_for_neighbors(
    pos_self: np.ndarray,
    vel_self: np.ndarray,
    neighbors: list,
    radius: float = 2.0,
    K: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute VO features for all neighbors.
    
    Args:
        pos_self: Position of self agent [x, y, z]
        vel_self: Velocity of self agent [vx, vy, vz]
        neighbors: List of neighbor dicts with 'pos' and 'vel' keys
        radius: Combined collision radius
        K: Fixed output length (if None, uses len(neighbors))
    
    Returns:
        collision_flags: Boolean array [K] indicating collision cone
        ttcs: Time-to-collision array [K]
        cpa_dists: Closest point of approach distances [K]
    """
    # Determine fixed output length
    if K is None:
        K = len(neighbors)
    
    # Initialize fixed-length arrays with safe defaults
    collision_flags = np.zeros(K, dtype=bool)
    ttcs = np.full(K, float('inf'))
    cpa_dists = np.full(K, float('inf'))
    
    # Early return with fixed-length arrays if no neighbors
    if len(neighbors) == 0:
        return collision_flags.astype(np.float32), ttcs, cpa_dists
    
    # Process each neighbor (up to K)
    num_to_process = min(len(neighbors), K)
    for i in range(num_to_process):
        neighbor = neighbors[i]
        if neighbor.get('is_obstacle', False):
            # Static obstacles - simpler collision check
            dist = neighbor['dist']
            if dist < radius:
                collision_flags[i] = True
                ttcs[i] = 0.0
                cpa_dists[i] = dist
            else:
                # Project velocity towards obstacle
                rel_pos = np.array(neighbor['pos']) - pos_self
                if np.linalg.norm(vel_self) > 1e-3:
                    # Check if moving towards obstacle
                    direction = vel_self / np.linalg.norm(vel_self)
                    proj = np.dot(rel_pos, direction)
                    if proj > 0:  # Moving towards
                        ttcs[i] = dist / np.linalg.norm(vel_self)
                        cpa_dists[i] = dist
        else:
            # Dynamic neighbor (other drone)
            pos_other = np.array(neighbor['pos'])
            vel_other = np.array(neighbor['vel'])
            
            is_in_cone, ttc, cpa = compute_velocity_obstacle(
                pos_self, vel_self, pos_other, vel_other, radius
            )
            
            collision_flags[i] = is_in_cone
            ttcs[i] = ttc
            cpa_dists[i] = cpa
    
    # Convert collision_flags to float32 for consistent observation types
    return collision_flags.astype(np.float32), ttcs, cpa_dists


def compute_collision_risk_reward(
    ttcs: np.ndarray,
    cpa_dists: np.ndarray,
    tau: float = 2.0,
    weight: float = 0.5
) -> float:
    """
    Compute smooth collision risk reward (gradient-based, not binary).
    
    Args:
        ttcs: Time-to-collision array for all neighbors
        cpa_dists: CPA distances for all neighbors
        tau: Time constant for exponential decay
        weight: Risk penalty weight
    
    Returns:
        collision_risk_penalty: Negative reward for collision risk
    """
    if len(ttcs) == 0:
        return 0.0
    
    # Filter approaching neighbors (ttc < threshold)
    approaching_mask = ttcs < 5.0
    
    if not np.any(approaching_mask):
        return 0.0
    
    # Exponential decay based on TTC
    risk_values = np.exp(-ttcs[approaching_mask] / tau)
    
    # Sum all risks (each neighbor contributes)
    total_risk = np.sum(risk_values)
    
    return -weight * total_risk
