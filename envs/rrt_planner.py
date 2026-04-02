"""
RRT* (Rapidly-exploring Random Tree Star) Path Planner
Implements informed-RRT* for global path planning in 3D environments
"""
import numpy as np
from typing import List, Tuple, Optional
import time


class Node:
    """Node in the RRT tree"""
    def __init__(self, position: np.ndarray, parent=None):
        self.position = np.array(position)  # [x, y, z]
        self.parent = parent
        self.cost = 0.0  # Cost from start
        
        
class RRTStarPlanner:
    """
    RRT* path planner for 3D obstacle-rich environments.
    Uses informed sampling after first solution found.
    """
    
    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        step_size: float = 2.0,
        goal_radius: float = 2.0,
        max_iterations: int = 1000,
        rewire_radius: float = 5.0
    ):
        """
        Initialize RRT* planner.
        
        Args:
            start: Start position [x, y, z]
            goal: Goal position [x, y, z]
            bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            step_size: Maximum edge length
            goal_radius: Distance threshold to goal
            max_iterations: Maximum planning iterations
            rewire_radius: Radius for rewiring in RRT*
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iterations = max_iterations
        self.rewire_radius = rewire_radius
        
        # Tree
        self.nodes = [Node(self.start)]
        self.goal_node = None
        
        # Obstacles (LiDAR point cloud)
        self.obstacles = []  # List of [x, y, z] positions
        self.obstacle_radius = 1.0  # Collision check radius
        
    def set_obstacles(self, obstacles: List[np.ndarray], radius: float = 1.0):
        """Set static obstacles from LiDAR scan"""
        self.obstacles = [np.array(obs) for obs in obstacles]
        self.obstacle_radius = radius
        
    def is_collision_free(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if line segment from pos1 to pos2 is collision-free"""
        # Sample points along the line
        num_samples = int(np.linalg.norm(pos2 - pos1) / 0.5) + 1
        samples = np.linspace(pos1, pos2, num_samples)
        
        for sample in samples:
            # Check against all obstacles
            for obs in self.obstacles:
                if np.linalg.norm(sample - obs) < self.obstacle_radius:
                    return False
        
        return True
    
    def sample_point(self, use_informed: bool = False) -> np.ndarray:
        """Sample random point in configuration space"""
        if use_informed and self.goal_node is not None:
            # Informed sampling: bias towards ellipsoid between start and goal
            c_best = self.goal_node.cost
            c_min = np.linalg.norm(self.goal - self.start)
            
            # Sample in ellipsoid (simplified: biased towards goal)
            if np.random.rand() < 0.5:
                # Sample near straight line from start to goal
                t = np.random.rand()
                point = self.start + t * (self.goal - self.start)
                # Add random offset
                point += np.random.randn(3) * (c_best - c_min) * 0.3
            else:
                # Uniform sampling
                point = self._uniform_sample()
        else:
            # Goal biasing: 10% chance to sample goal
            if np.random.rand() < 0.1:
                return self.goal.copy()
            
            point = self._uniform_sample()
        
        # Clip to bounds
        point[0] = np.clip(point[0], self.bounds[0][0], self.bounds[0][1])
        point[1] = np.clip(point[1], self.bounds[1][0], self.bounds[1][1])
        point[2] = np.clip(point[2], self.bounds[2][0], self.bounds[2][1])
        
        return point
    
    def _uniform_sample(self) -> np.ndarray:
        """Uniform random sample in bounds"""
        x = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
        z = np.random.uniform(self.bounds[2][0], self.bounds[2][1])
        return np.array([x, y, z])
    
    def nearest_node(self, point: np.ndarray) -> Node:
        """Find nearest node in tree to given point"""
        min_dist = float('inf')
        nearest = self.nodes[0]
        
        for node in self.nodes:
            dist = np.linalg.norm(node.position - point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Steer from from_pos towards to_pos with step_size limit"""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        
        if dist <= self.step_size:
            return to_pos
        
        return from_pos + (direction / dist) * self.step_size
    
    def near_nodes(self, position: np.ndarray) -> List[Node]:
        """Find all nodes within rewire_radius of position"""
        near = []
        for node in self.nodes:
            if np.linalg.norm(node.position - position) < self.rewire_radius:
                near.append(node)
        return near
    
    def plan(self, timeout: float = 5.0) -> Optional[List[np.ndarray]]:
        """
        Run RRT* planning algorithm.
        
        Args:
            timeout: Maximum planning time in seconds
        
        Returns:
            List of waypoints [N, 3] from start to goal, or None if failed
        """
        start_time = time.time()
        use_informed = False
        
        for i in range(self.max_iterations):
            if time.time() - start_time > timeout:
                break
            
            # Sample random point
            rand_point = self.sample_point(use_informed=use_informed)
            
            # Find nearest node
            nearest = self.nearest_node(rand_point)
            
            # Steer towards sampled point
            new_pos = self.steer(nearest.position, rand_point)
            
            # Check collision
            if not self.is_collision_free(nearest.position, new_pos):
                continue
            
            # Find near nodes for RRT* rewiring
            near_nodes = self.near_nodes(new_pos)
            
            # Choose parent with minimum cost
            min_cost = nearest.cost + np.linalg.norm(new_pos - nearest.position)
            best_parent = nearest
            
            for near_node in near_nodes:
                potential_cost = near_node.cost + np.linalg.norm(new_pos - near_node.position)
                if potential_cost < min_cost and self.is_collision_free(near_node.position, new_pos):
                    min_cost = potential_cost
                    best_parent = near_node
            
            # Create new node
            new_node = Node(new_pos, parent=best_parent)
            new_node.cost = min_cost
            self.nodes.append(new_node)
            
            # Rewire tree
            for near_node in near_nodes:
                new_cost = new_node.cost + np.linalg.norm(near_node.position - new_pos)
                if new_cost < near_node.cost and self.is_collision_free(new_pos, near_node.position):
                    near_node.parent = new_node
                    near_node.cost = new_cost
            
            # Check if reached goal
            if np.linalg.norm(new_pos - self.goal) < self.goal_radius:
                if self.goal_node is None:
                    # First solution found - switch to informed sampling
                    self.goal_node = new_node
                    use_informed = True
                elif new_node.cost < self.goal_node.cost:
                    # Better solution found
                    self.goal_node = new_node
        
        # Extract path
        if self.goal_node is None:
            return None  # No path found
        
        path = []
        current = self.goal_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()
        return path
    
    def simplify_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Remove redundant waypoints using line-of-sight"""
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to skip ahead as far as possible
            for i in range(len(path) - 1, current_idx, -1):
                if self.is_collision_free(path[current_idx], path[i]):
                    simplified.append(path[i])
                    current_idx = i
                    break
        
        return simplified
