import numpy as np
import time
from typing import Dict, List, Tuple


class MockAirSimClientWrapper:
    """
    Mock AirSim client for smoke testing without requiring Unreal Engine.
    Implements simple deterministic kinematics for drone simulation.
    """
    
    def __init__(self, num_agents=1, ip="127.0.0.1", port=41451):
        """
        Initialize mock client with deterministic state.
        
        Args:
            num_agents: Number of drones to simulate
            ip: Ignored (for interface compatibility)
            port: Ignored (for interface compatibility)
        """
        self.num_agents = num_agents
        self.drone_names = [f"Drone{i}" for i in range(num_agents)]
        
        # Internal state storage: {drone_name: state_dict}
        self.states = {}
        self.start_time = time.time()
        self._initialize_states()
    
    def get_sim_time(self):
        """Get mock simulation time in nanoseconds."""
        return int((time.time() - self.start_time) * 1e9)

    def _initialize_states(self):
        """Initialize all drones at origin with zero velocity."""
        for i, name in enumerate(self.drone_names):
            # Spread drones slightly to avoid exact overlap
            offset = i * 2.0
            self.states[name] = {
                'pos': np.array([offset, 0.0, 0.0], dtype=np.float32),
                'vel': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'yaw': 0.0,
                'collision': False
            }
    
    def verify_connection(self):
        """Mock connection verification - always returns True."""
        return True
    
    def reset(self, spawn_positions=None):
        """
        Reset all drones to initial states.
        
        Args:
            spawn_positions: Optional dict {agent_name: (x_enu, y_enu, z_enu)} for initial positions
        """
        self._initialize_states()
        
        # If spawn positions provided, override default positions
        if spawn_positions:
            for name, pos in spawn_positions.items():
                if name in self.states:
                    self.states[name]['pos'] = np.array(pos, dtype=np.float32)
    
    def set_drone_poses(self, poses):
        """
        Set specific drone poses (optional, for curriculum).
        
        Args:
            poses: dict {agent_id: (x_enu, y_enu, z_enu, yaw_enu)}
        """
        for name, pose in poses.items():
            if name in self.states:
                x, y, z, yaw = pose
                self.states[name]['pos'] = np.array([x, y, z], dtype=np.float32)
                self.states[name]['yaw'] = float(yaw)
    
    def get_drone_states(self) -> Dict[str, Dict]:
        """
        Get current state of all drones in ENU frame.
        
        Returns:
            dict: {agent_id: {pos: [x,y,z], vel: [vx,vy,vz], yaw: float, collision: bool}}
        """
        # Return deep copy to prevent external modification
        return {
            name: {
                'pos': self.states[name]['pos'].copy().tolist(),
                'vel': self.states[name]['vel'].copy().tolist(),
                'yaw': float(self.states[name]['yaw']),
                'collision': bool(self.states[name]['collision'])
            }
            for name in self.drone_names
        }
    
    
    
    def get_obstacles(self):
        """Mock obstacles for infrastructure-aware safe spawning."""
        # Return a list of mock obstacles or empty list
        # Format: [{'pos': [x, y, z], 'scale': [sx, sy, sz]}]
        return []

    def get_nearest_obstacles(self, vehicle_name, drone_pos_enu, num_obstacles=5, min_altitude=1.0, all_drone_positions=None):
        """
        Mock obstacle detection - returns empty list (no obstacles in mock environment).
        
        Args:
            vehicle_name: Name of the drone
            drone_pos_enu: Current position [x, y, z] in ENU
            num_obstacles: Maximum number of obstacles to return
            min_altitude: Minimum altitude for filtering ground
            all_drone_positions: List of all drone positions to filter out
        
        Returns:
            Empty list (no static obstacles in mock mode)
        """
        # Mock mode: no static obstacles, only other drones (handled separately)
        return []
    
    def apply_actions(self, actions: Dict[str, np.ndarray], dt: float, action_frequency: int = 1) -> List:
        """
        Apply velocity commands and integrate kinematics.
        
        Args:
            actions: dict {agent_id: (vx_body, vy_body, vz_world, yaw_rate)}
            dt: Time step in seconds
            
        Returns:
            Empty list (no futures in mock mode)
        """
        for name, action in actions.items():
            if name not in self.states:
                continue
            
            vx_body, vy_body, vz_enu, yaw_rate = action
            
            # Get current state
            state = self.states[name]
            yaw = state['yaw']
            
            # Convert body-frame velocities to world frame (ENU)
            # Body frame: X=forward, Y=left, Z=up
            # Rotation matrix for yaw (around Z-axis)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # Transform body velocities to ENU
            vx_enu = cos_yaw * vx_body - sin_yaw * vy_body
            vy_enu = sin_yaw * vx_body + cos_yaw * vy_body
            # vz_enu is already in world frame
            
            # Update velocity
            state['vel'] = np.array([vx_enu, vy_enu, vz_enu], dtype=np.float32)
            
            # Integrate position: p += v * dt
            state['pos'] += state['vel'] * dt
            
            # Integrate yaw: yaw += yaw_rate * dt
            state['yaw'] += yaw_rate * dt
            
            # Normalize yaw to [-pi, pi]
            state['yaw'] = np.arctan2(np.sin(state['yaw']), np.cos(state['yaw']))
            
            # Collision always False in mock (deterministic)
            state['collision'] = False
        
        # Return empty list (no async futures in mock mode)
        return []
