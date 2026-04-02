import functools
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np
import yaml
import json
import time
from loguru import logger

from .airsim_client import AirSimClientWrapper
from .mock_airsim_client import MockAirSimClientWrapper
from .utils_observation import build_observation, get_observation_space_shape
from .utils_reward import compute_reward
from .utils_logging import BatchLogger
from .frames import ned_to_enu_pos, ned_to_enu_vel
from .pid_controller import UAVPIDController
from .rrt_planner import RRTStarPlanner  # PHASE 1: Global planner

class UniversalUAVEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "universal_uav_v0"}

    def __init__(self, config_path="config/env.yaml", smoke_test=False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.initial_num_agents = self.config['env']['num_agents']
        self.agent_name_prefix = self.config['env']['agent_name_prefix']
        self.possible_agents = [f"{self.agent_name_prefix}{i}" for i in range(self.initial_num_agents)]
        self.agents = self.possible_agents[:]
        self.smoke_test = smoke_test
        
        # AirSim Client (real or mock)
        ip = self.config['env']['ip']
        port = self.config['env']['port']
        if not smoke_test:
            self.airsim_client = AirSimClientWrapper(num_agents=self.initial_num_agents, ip=ip, port=port)
        else:
            self.airsim_client = MockAirSimClientWrapper(num_agents=self.initial_num_agents, ip=ip, port=port)
        
        # Observation Space
        obs_shape = get_observation_space_shape(self.config['observation'])
        # Use -inf to inf for now, or scaled. Tanh scales to -1..1.
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # [vx_body, vy_body, vz_enu, yaw_rate]
        # vx_body: forward velocity in body frame (m/s), range [-10, 10]
        # vy_body: lateral velocity in body frame (m/s), range [-5, 5]
        # vz_enu: vertical velocity in ENU world frame (m/s, up is positive), range [-5, 5]
        # yaw_rate: yaw angular velocity (rad/s), range [-2, 2]
        # These are CONTINUOUS actions        
        # Action Space
        # PHASE 1: Support simplified 3-DOF action space [speed_cmd, yaw_rate_cmd, climb_rate_cmd]
        if self.config['env'].get('use_simplified_actions', False):
            # Simplified: [speed (0-10 m/s), yaw_rate (-2 to 2 rad/s), climb_rate (-2 to 2 m/s)]
            self.action_spaces = {
                agent: spaces.Box(
                    low=np.array([0.0, -2.0, -2.0]),
                    high=np.array([10.0, 2.0, 2.0]),
                    dtype=np.float32
                ) for agent in self.possible_agents
            }
        else:
            # Original: [dx, dy, dz, dyaw] velocity setpoints
            self.action_spaces = {
                agent: spaces.Box(
                    low=np.array([-10.0, -5.0, -5.0, -2.0]),
                    high=np.array([10.0, 5.0, 5.0, 2.0]),
                    dtype=np.float32
                ) for agent in self.possible_agents
            }
        

        # Internal State
        self.dt = self.config['env']['sim_dt']
        self.max_steps = self.config['env']['max_steps']
        
        # Initialize state buffers
        self.goals = {}
        self.prev_states = {}
        self.prev_dist_to_goal = {}
        self.prev_min_separation = {}
        self.cumulative_rewards = {}  # Track cumulative reward per agent
        
        # Collision tracking to prevent repeated penalties
        self.collision_state = {}  # Track if agent is currently in collision
        self.collision_grace_steps = {}
        self.success_agents = set()  # Track agents that have reached goal to keep them alive (hovering)  # Grace period after spawn/reset
        
        # PHASE 1: Simplified action space mode
        self.use_simplified_actions = self.config['env'].get('use_simplified_actions', False)
        
        # PHASE 1: RRT* Planner state
        self.planners = {}  # RRT* planner per agent
        self.planned_paths = {}  # Waypoint lists per agent
        self.current_waypoint_idx = {}  # Current waypoint index per agent
        self.use_global_planner = self.config['env'].get('use_global_planner', False)
        
        # Curriculum Learning State
        self.curriculum_config = self.config['env'].get('curriculum', {})
        self.curriculum_mode = self.curriculum_config.get('mode', 'mixed')
        self.curriculum_level = 0  # 0 = easiest, will increment over time
        self.episode_count = 0
        self.total_step_count = 0 # Track total steps for step-based curriculum
        self.episode_success_count = 0  # Track episode-level successes (not per-agent)
        self.episode_success_count = 0  # Track episode-level successes (not per-agent)
        self.current_episode_has_success = False  # Flag for current episode

        # Override max_steps for smoke test to ensure quick episode completion
        if smoke_test:
            self.max_steps = 50
        
        
        self.step_count = 0
        
        # Logging
        self.logger = None
        self.log_dir = self.config.get('logging', {}).get('log_dir', None)
        # Check config safely
        if self.config.get('logging', {}).get('enable', False) and self.log_dir:
            self.logger = BatchLogger(self.log_dir)
        
        # PID Controllers for hover stabilization (one per agent)
        self.pid_controllers = {
            agent: UAVPIDController(dt=self.dt) for agent in self.possible_agents
        }
        self.hover_targets = {}  # Hover positions for each agent
        self.use_pid_assist = self.config['env'].get('use_pid_assist', True)
        self.pid_blend_weight = self.config['env'].get('pid_blend_weight', 0.3)  # 30% PID, 70% RL
        
        # --- COOPERATIVE RL STATE ---
        self.message_history = {agent: [] for agent in self.possible_agents}
        self.last_delivered_messages = {agent: {} for agent in self.possible_agents}
        self.comm_config = self.config['env'].get('communication', {})
        self.coop_config = self.config['env'].get('cooperation', {})
        
        # Cumulative action control
        self.action_frequency = self.config['env'].get('action_frequency', 10)  # Apply actions every N steps
        self.cumulative_actions = {}  # Accumulated actions for each agent
        self.last_sent_actions = {}  # Last actions sent to AirSim
        
        # Load static infrastructure map for safe spawning
        logger.info("Querying environment for static obstacles...")
        self.internal_obstacles = self.airsim_client.get_obstacles()
        
        # Adjust maximum altitude based on tallest obstacle
        self.max_obstacle_height = 0.0
        if self.internal_obstacles:
            for obs in self.internal_obstacles:
                # Top z = center z + height/2
                # ENU: z is up. Scl z is dimension.
                # Assuming simple bounding box logic:
                # Standard cube is 100x100x100 Unreal Units. 
                # Scale 1.0 = 100 UU = 1m? Or Scale 1.0 = 1m?
                # AirSim standard: Scale is multiplier. If base mesh is 1m, scale 10 is 10m.
                # Let's assume extent from scale.
                z_top = obs['pos'][2] + (obs['scale'][2] * 100.0 / 2.0 / 100.0) # If 1UU=1cm, then 100UU=1m
                # Actually, usually 1 UU = 1cm. So 100 units = 1m.
                # If scale is 10, that's 10m.
                # Let's verify with log later.
                
                # Simplified: Z + Scale*0.5 (assuming base is ~1m) * 1.0
                # Wait, "TemplateCube_48" scale in log was (10, 10, 5).
                # If that's meters, height is 5m.
                h = obs['pos'][2] + obs['scale'][2] * 0.5 * 100.0
                # Wait, earlier log said: Scale: <Vector3r> { x_val: 10.0, y_val: 10.0, z_val: 5.0 }
                # The prompt said "check if we can access the outline...".
                # If scale is raw Unreal scale, usually 1.0 = original mesh size.
                # If using "1x1x1m Cube", then Scale 5 = 5m.
                # The log showed Z pos -1.48 and scale Z 5.0. 
                # If ENU Z is up, and ground is 0. 
                # If it's 5m tall, centered at Z=?, 
                # Let's trust the logic: Center + Scale/2.
                # For safety, treat scale as meters for now (as is common in these sims).
                top = obs['pos'][2] + (obs['scale'][2] / 2.0)
                if top > self.max_obstacle_height:
                    self.max_obstacle_height = top
                    
        # Update config with safe altitude
        safe_alt_max = self.max_obstacle_height + 5.0
        self.config['reward']['altitude_max'] = max(self.config['reward'].get('altitude_max', 20.0), safe_alt_max)
        logger.info(f"Updated altitude_max to {self.config['reward']['altitude_max']:.2f}m based on infrastructure (max height: {self.max_obstacle_height:.2f}m)")
        
        # Initialize timestamp for synchronization (if not smoke test)
        self.last_sim_time_nanos = 0
        self.clock_speed = 1.0 # Default
        
        # Load ClockSpeed just for logging purposes
        try:
            with open("config/airsim_settings.json", "r") as f:
                settings = json.load(f)
                self.clock_speed = settings.get("ClockSpeed", 1.0)
        except Exception:
            pass
            
        if not self.smoke_test:
            self.last_sim_time_nanos = self.airsim_client.get_sim_time()

    def _update_curriculum(self):
        """Update curriculum level based on step count or success rate."""
        max_level = self.curriculum_config.get('max_level', 3)
        update_interval = self.curriculum_config.get('update_interval_steps', 100000)
        
        # 1. Step-based Check (Deterministic)
        if self.curriculum_mode in ['step', 'mixed']:
            # Calculate level based on total steps
            step_level = min(self.total_step_count // update_interval, max_level)
            
            if step_level > self.curriculum_level:
                self.curriculum_level = step_level
                logger.info(f"Curriculum STEP-UP to level {self.curriculum_level} (total_steps: {self.total_step_count})")
                return # Priority to step-based progress
        
        # 2. Success-based Check (Adaptive)
        # Only check periodically to avoid noise
        if self.episode_count % 50 == 0 and self.episode_count > 0:
            success_rate = self.episode_success_count / 50.0
            
            # Increase difficulty
            if success_rate > 0.6 and self.curriculum_level < max_level:
                self.curriculum_level += 1
                logger.info(f"Curriculum SUCCESS-UP to level {self.curriculum_level} (success rate: {success_rate:.2%})")
                self.episode_success_count = 0 
            
            # Decrease difficulty (Adaptive fallback), but not below step-based floor if mixed
            elif success_rate < 0.2 and self.curriculum_level > 0:
                # In mixed mode, don't drop below the step-based level
                min_level = 0
                if self.curriculum_mode == 'mixed':
                    min_level = min(self.total_step_count // update_interval, max_level)
                
                if self.curriculum_level > min_level:
                    self.curriculum_level -= 1
                    logger.info(f"Curriculum DOWN to level {self.curriculum_level} (success rate: {success_rate:.2%})")
            
            # Reset counter after check
            if self.episode_count % 50 == 0:
                 self.episode_success_count = 0

    def _is_pos_safe(self, pos, margin=2.0):
        """Check if position is inside any static obstacle."""
        px, py, pz = pos
        for obs in self.internal_obstacles:
            c = obs['pos']
            s = obs['scale']
            # Assume base mesh is 1m x 1m x 1m unit cube centered at origin
            # Extents = scale / 2
            dx = s[0] / 2.0 + margin
            dy = s[1] / 2.0 + margin
            dz = s[2] / 2.0 + margin
            
            if (abs(px - c[0]) < dx and 
                abs(py - c[1]) < dy and 
                abs(pz - c[2]) < dz):
                return False
        return True
            
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Robust safety: Start with fresh dicts
        self.goals = {}
        self.prev_states = {}
        self.prev_dist_to_goal = {}
        self.prev_min_separation = {}
        self.cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.episode_count += 1
        
        # Track episode success from previous episode
        if hasattr(self, 'current_episode_has_success') and self.current_episode_has_success:
            self.episode_success_count += 1
        self.current_episode_has_success = False  # Reset for new episode
        self.success_agents = set()  # Reset persistence tracking
        
        # CURRICULUM: Determine difficulty based on curriculum level
        self._update_curriculum()
        
        # Get parameters for current level
        levels = self.curriculum_config.get('levels', {
            0: {'dist': [15, 25], 'alt': [8, 12]},
            1: {'dist': [25, 40], 'alt': [5, 15]},
            2: {'dist': [40, 60], 'alt': [5, 20]}
        })
        
        # Fallback if level not in config
        current_params = levels.get(self.curriculum_level, levels.get(len(levels)-1))
        
        goal_dist_range = current_params['dist']
        altitude_range = current_params['alt']
        
        # Adjust spacing/spawn based on difficulty
        if self.curriculum_level == 0:
            spacing = 15.0
            spawn_altitude = 10.0
        elif self.curriculum_level == 1:
            spacing = 12.0
            spawn_altitude = 8.0
        else:
            spacing = 10.0
            spawn_altitude = np.random.uniform(5, 12)
        
        # Create spawn positions with minimum separation to avoid initial collisions
        spawn_positions = {}
        
        # Global random offset for the fleet to explore different starting areas
        # Keep within safe bounds (e.g. -20m to +20m)
        fleet_offset_x = np.random.uniform(-20.0, 20.0)
        fleet_offset_y = np.random.uniform(-20.0, 20.0)
        
        for i, agent in enumerate(self.agents):
            row = i // 3
            col = i % 3
            spawn_x = (col * spacing) + fleet_offset_x # ENU x
            spawn_y = (row * spacing) + fleet_offset_y # ENU y
            spawn_z = spawn_altitude  # Start at safe altitude (ENU z-up is positive)
            spawn_positions[agent] = (spawn_x, spawn_y, spawn_z)
        
        # Reset AirSim with spawn positions
        self.airsim_client.reset(spawn_positions=spawn_positions)
        
        # Initialize collision tracking with grace period
        # Grace period = action_frequency (drones can't move until first action sent)
        collision_grace_period = self.config['env'].get('action_frequency', 10)
        for agent in self.agents:
            self.collision_state[agent] = False
            self.collision_grace_steps[agent] = collision_grace_period
        
        # Initialize hover targets, reset PID controllers, and reset cumulative actions
        for agent in self.agents:
            self.hover_targets[agent] = np.array(spawn_positions[agent])
            self.pid_controllers[agent].reset()
            self.cumulative_actions[agent] = np.zeros(4, dtype=np.float32)  # [vx, vy, vz, yaw_rate]
            self.last_sent_actions[agent] = np.zeros(4, dtype=np.float32)
        
        # Sample Goals with curriculum difficulty
        # Sample Goals with curriculum difficulty
        goal_min_dist = 10.0 # Minimum separation between goals
        
        for agent in self.agents:
            spawn_pos = np.array(spawn_positions[agent])
            
            valid_goal = False
            attempts = 0
            while not valid_goal and attempts < 20:
                attempts += 1
                
                # Generate goal at curriculum-appropriate distance
                goal_distance = np.random.uniform(*goal_dist_range)
                
                # Random direction in horizontal plane
                angle = np.random.uniform(0, 2 * np.pi)
                
                # Goal position relative to spawn
                gx = spawn_pos[0] + goal_distance * np.cos(angle)
                gy = spawn_pos[1] + goal_distance * np.sin(angle)
                gz = np.random.uniform(*altitude_range)
                
                candidate_goal = np.array([gx, gy, gz])
                
                # Check separation from other existing goals
                too_close = False
                for other_agent, other_goal in self.goals.items():
                    if np.linalg.norm(candidate_goal - other_goal) < goal_min_dist:
                        too_close = True
                        break
                
                if not too_close:
                    # Check against static infrastructure
                    if self._is_pos_safe(candidate_goal):
                        valid_goal = True
                        self.goals[agent] = candidate_goal
            
            # If valid goal not found in attempts, force it anyway (fallback)
            if agent not in self.goals:
                 self.goals[agent] = candidate_goal
        
        # Get initial states to compute actual distances
        states = self.airsim_client.get_drone_states()
        
        # Initialize prev_states and derivative buffers with actual initial values
        for agent in self.agents:
            gx, gy, gz = self.goals[agent]
            dist0 = np.linalg.norm(np.array(states[agent]["pos"]) - np.array([gx, gy, gz]))
            
            # BUGFIX: Initialize            # Store current state for next step's derivatives
            # CRITICAL: Store CONVERTED actions (4-DOF) not raw actions (3-DOF) for observation consistency
            self.prev_states[agent] = {
                "dist_to_goal": float(dist0),
                "prev_action": np.zeros(4, dtype=np.float32),  # Always 4-DOF after conversion, initialized to zero
                "prev_vel": np.array(states[agent]["vel"], dtype=np.float32),
                "was_colliding": states[agent]['is_colliding'] if 'is_colliding' in states[agent] else False
            }
            
            # Initialize derivative buffers
            self.prev_dist_to_goal[agent] = float(dist0)
            self.prev_min_separation[agent] = float('inf')  # No separation data yet
            
            # PHASE 1: RRT* Global Path Planning
            if self.use_global_planner:
                planner = RRTStarPlanner(
                    start=states[agent]["pos"],
                    goal=self.goals[agent],
                    bounds=((-100, 100), (-100, 100), (3, 25)),  # Environment bounds
                    step_size=5.0,
                    goal_radius=2.0,
                    max_iterations=500,
                    rewire_radius=10.0
                )
                
                # Get static obstacles from LiDAR for path planning
                # For now, plan with empty obstacle set (will be improved in later phases)
                # In practice, we'd use a voxel map or obstacle database
                path = planner.plan(timeout=2.0)  # 2 second planning budget
                
                if path is not None:
                    # Simplify path
                    simplified_path = planner.simplify_path(path)
                    self.planned_paths[agent] = simplified_path
                    self.current_waypoint_idx[agent] = 0
                else:
                    # Fallback: direct line to goal
                    self.planned_paths[agent] = [states[agent]["pos"], self.goals[agent]]
                    self.current_waypoint_idx[agent] = 0
            
        # Initialize waypoint state for all agents (even if not using planner)
        for agent in self.agents:
            if agent not in self.current_waypoint_idx:
                self.current_waypoint_idx[agent] = 0
                # Fallback path: direct to goal
                self.planned_paths[agent] = [states[agent]["pos"], self.goals[agent]]
            
        # Get Obs
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        
        if self.logger:
            import uuid
            eid = str(uuid.uuid4())[:8]
            self.logger.start_episode(eid)
            
        return observations, infos

    def step(self, actions):
        # Capture start of step logic (wall time for logging/timeouts only)
        t0 = time.time()
        
        # Increment step count FIRST
        self.step_count += 1
        self.total_step_count += 1 # Global step counter for curriculum
        
        # Apply actions
        # actions is dict {agent: action_vec}
        # Filter dead agents
        active_actions = {a: act for a, act in actions.items() if a in self.agents}
        
        # Get current states for PID assistance
        current_states = self.airsim_client.get_drone_states()
        
        # PHASE 1: Convert simplified actions to velocity commands if using simplified mode
        if self.use_simplified_actions:
            converted_actions = {}
            for agent in active_actions:
                # Simplified action: [speed, yaw_rate, climb_rate]
                speed, yaw_rate_cmd, climb_rate = active_actions[agent]
                
                # Get current yaw
                yaw = current_states[agent]['yaw']
                
                # Convert speed to body-frame velocity
                vx_body = speed  # Forward speed
                vy_body = 0.0    # No lateral movement in simplified mode
                vz_enu = climb_rate  # Climb rate (ENU frame)
                
                # Convert to action format [vx_body, vy_body, vz_enu, yaw_rate]
                converted_actions[agent] = np.array([vx_body, vy_body, vz_enu, yaw_rate_cmd], dtype=np.float32)
            
            active_actions = converted_actions
        
        # Store converted 4-DOF actions for prev_action tracking (CRITICAL for observation dimensionality)
        actions_for_observation = active_actions.copy()
        
        # Blend RL actions with PID stabilization to prevent falling
        if self.use_pid_assist:
            blended_actions = {}
            for agent in active_actions:
                rl_action = np.array(active_actions[agent])
                
                # Get current position and yaw
                current_pos = np.array(current_states[agent]['pos'])
                current_yaw = current_states[agent]['yaw']
                
                # Compute PID control to maintain hover at spawn position
                pid_vx, pid_vy, pid_vz, pid_yaw_rate = self.pid_controllers[agent].compute_hover_control(
                    current_pos=current_pos,
                    hover_pos=self.hover_targets[agent],
                    current_yaw=current_yaw
                )
                
                # Blend: PID provides baseline stabilization, RL adds learned behavior
                # During early training, this prevents catastrophic falls
                blend_weight = self.pid_blend_weight
                blended_action = (1 - blend_weight) * rl_action + blend_weight * np.array([pid_vx, pid_vy, pid_vz, pid_yaw_rate])
                
                blended_actions[agent] = blended_action
            
            active_actions = blended_actions
        
        # Curriculum 2D mode: lock vz to 0 if enabled (Section 3)
        if self.config['env'].get('curriculum_2d_mode', False):
            for agent in active_actions:
                active_actions[agent] = np.array([
                    active_actions[agent][0],  # vx_body
                    active_actions[agent][1],  # vy_body
                    0.0,                        # vz_enu locked to 0
                    active_actions[agent][3]   # yaw_rate
                ])
        
        # No cumulative actions - direct synchronous control
        # Apply limits (Deadzone and Clipping)
        min_threshold = self.config['env'].get('min_action_threshold', 0.1)
        max_magnitude = self.config['env'].get('max_action_magnitude', 10.0)
        
        actions_to_send = {}
        for agent in active_actions:
            # Override action if agent already succeeded (force hover)
            if agent in self.success_agents:
                # Hover: speed=0, yaw_rate=0, climb_rate=0
                # This ensures they stay at goal for visualization
                # Use raw active_actions[agent] to match shape
                actions_to_send[agent] = np.zeros_like(active_actions[agent], dtype=np.float32)
                continue

            raw_action = np.array(active_actions[agent], dtype=np.float32)
            
            # Deadzone: if magnitude is too small, zero it out (avoids micro-jitter)
            if np.linalg.norm(raw_action) < min_threshold:
                final_action = np.zeros(4, dtype=np.float32)
            else:
                final_action = raw_action
                
                # Clipping: Clamp maximum velocity magnitude
                # Note: simplified actions are already bounded by action space, but this is a safety layer
                # Only clip linear velocity (first 3 components), leave yaw rate (index 3)
                v_norm = np.linalg.norm(final_action[:3])
                if v_norm > max_magnitude:
                     final_action[:3] = final_action[:3] * (max_magnitude / v_norm)
            
            actions_to_send[agent] = final_action
            self.last_sent_actions[agent] = final_action.copy()
            # Reset cumulative buffer just in case (though unused now)
            self.cumulative_actions[agent] = np.zeros(4, dtype=np.float32)
        
        # --- COOPERATIVE ARBITRATION LAYER ---
        # Before sending to AirSim, apply rule-based bias for CD&R
        if self.coop_config.get('use_priority_negotiation', True):
            actions_to_send = self._apply_cooperative_arbitration(actions_to_send, current_states)

        # Apply actions asynchronously (futures execute in background)
        futures = self.airsim_client.apply_actions(actions_to_send, self.dt, self.action_frequency)
        
        # Get updated states after action execution
        current_states = self.airsim_client.get_drone_states()
        
        t1 = time.time() # RPC End
        
        # === LIDAR & OBSTACLE SENSING (Added for Soft Obstacle Reward) ===
        # Fetch LiDAR data for all agents (Centralized for efficiency)
        # This powers both the 'r_obs_proximity' reward and the observation vector
        all_drone_positions = [np.array(current_states[a]['pos']) for a in self.agents]
        min_altitude = 1.0
        num_lidar_obstacles = self.config['observation'].get('K', 5)

        for agent in self.agents:
            lidar_obstacles = self.airsim_client.get_nearest_obstacles(
                agent, 
                current_states[agent]['pos'],
                num_obstacles=num_lidar_obstacles,
                min_altitude=min_altitude,
                all_drone_positions=all_drone_positions
            )
            current_states[agent]['lidar_obstacles'] = lidar_obstacles
            
            # Extract rays for reward function (just the distances)
            # utils_reward.py uses np.min(obstacle_rays)
            if lidar_obstacles:
                dists = [o['dist'] for o in lidar_obstacles]
                current_states[agent]['obstacle_rays'] = np.array(dists)
            else:
                 current_states[agent]['obstacle_rays'] = np.array([50.0])
        
        # === SWARM-LEVEL STATISTICS (Section 4.1) ===
        # Compute once per step and inject into agent states for reward/obs
        positions = np.array([current_states[a]['pos'] for a in self.agents])
        velocities = np.array([current_states[a]['vel'] for a in self.agents])
        
        if len(self.agents) > 0:
            # Swarm centroid
            centroid_pos = np.mean(positions, axis=0)
            centroid_vel = np.mean(velocities, axis=0)
        else:
            centroid_pos = np.zeros(3)
            centroid_vel = np.zeros(3)
        
        # Calculate min separations for logging/reward
        # O(N^2) but N is small
        min_seps = {}
        ttc_mins = {}  # Time-to-collision (Section 4.1)
        for a in self.agents:
            p_a = np.array(current_states[a]['pos'])
            v_a = np.array(current_states[a]['vel'])
            d_min = float('inf')
            ttc_min = float('inf')
            
            for b in self.agents:
                if a == b: continue
                p_b = np.array(current_states[b]['pos'])
                v_b = np.array(current_states[b]['vel'])
                
                # Minimum separation
                dist = np.linalg.norm(p_a - p_b)
                if dist < d_min:
                    d_min = dist
                
                # Time-to-collision (predictive safety, Section 4.1)
                rel_pos = p_b - p_a
                rel_vel = v_b - v_a
                closing_speed = -np.dot(rel_pos, rel_vel) / (dist + 1e-6)
                
                if closing_speed > 0:  # Approaching
                    ttc = dist / closing_speed
                    if ttc < ttc_min:
                        ttc_min = ttc
            
            min_seps[a] = d_min
            ttc_mins[a] = ttc_min
            
            # Inject swarm stats into agent state for reward/obs
            current_states[a]['centroid_pos'] = centroid_pos
            current_states[a]['centroid_vel'] = centroid_vel
            current_states[a]['ttc_min'] = ttc_min
        
        # Compute derivative features and inject into states
        for agent in self.agents:
            st = current_states[agent]
            
            # Current distance to goal
            current_dist = np.linalg.norm(np.array(st['pos']) - self.goals[agent])
            
            # Progress rate: positive when getting closer
            prev_dist = self.prev_dist_to_goal.get(agent, current_dist)
            progress_rate = prev_dist - current_dist  # Positive = progress
            
            # Safety rate: positive when separation increasing
            current_min_sep = min_seps.get(agent, float('inf'))
            prev_min_sep = self.prev_min_separation.get(agent, current_min_sep)
            safety_rate = current_min_sep - prev_min_sep  # Positive = safer
            
            # Inject into state dict for observation building
            st['progress_rate'] = progress_rate
            st['safety_rate'] = safety_rate
            
            # Update buffers for next step
            self.prev_dist_to_goal[agent] = current_dist
            self.prev_min_separation[agent] = current_min_sep
        
        # --- TEAM LEVEL PROGRESS ---
        active_progress = [current_states[a]['progress_rate'] for a in self.agents]
        if active_progress:
            team_avg_progress = np.mean(active_progress)
        else:
            team_avg_progress = 0.0
        for agent in self.agents:
            current_states[agent]['team_avg_progress'] = team_avg_progress

        # Observations (now includes derivative features)
        # BUGFIX: Pass cached states to avoid re-fetching and losing computed features
        observations = self._get_obs(cached_states=current_states)
        
        # Rewards & Terminations
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        # Use the current_states we already have (with derivative features)
        # Calculate min separations for logging/reward
        # O(N^2) but N is small
        min_seps = {}
        for a in self.agents:
            p_a = np.array(current_states[a]['pos'])
            d_min = float('inf')
            for b in self.agents:
                if a == b: continue
                p_b = np.array(current_states[b]['pos'])
                dist = np.linalg.norm(p_a - p_b)
                if dist < d_min:
                    d_min = dist
            min_seps[a] = min(d_min, 200.0)  # Clip to finite sensor horizon
        
        for agent in self.agents:
            info = {}
            info['sep_min'] = min_seps.get(agent, float('inf'))
            info['goal_pos'] = self.goals[agent]
            # Add collision object name for debugging infrastructure impact
            info['collision_name'] = current_states[agent].get('collision_name', None)
            info['collision'] = current_states[agent].get('collision', False)
            
            # Add cumulative reward to info for catastrophic check
            info['cumulative_reward'] = self.cumulative_rewards.get(agent, 0.0)
            
            # Add current action and previous action for action cost penalties
            info['action'] = active_actions.get(agent, np.zeros(4))
            
            # State dict for reward
            st = current_states[agent]
            
            # Prepare state dict for reward computation
            st_reward = current_states[agent].copy()
            
            # Override collision flag during grace period
            if self.collision_grace_steps[agent] > 0:
                st_reward['collision'] = False  # Ignore collisions during grace period
            
            rew, term, trunc = compute_reward(
                st_reward, self.goals[agent], 
                self.prev_states.get(agent, {}), 
                self.config['reward'], 
                info
            )
            
            # Update cumulative reward
            self.cumulative_rewards[agent] = self.cumulative_rewards.get(agent, 0.0) + rew
            
            # Truncation by time
            if self.step_count >= self.max_steps:
                trunc = True
            
            # CRITICAL: If collision, force explicit termination in the environment dicts too
            if info.get('outcome') in ['collision', 'stuck_collision']:
                term = True
            
            rewards[agent] = rew
            terminations[agent] = term
            truncations[agent] = trunc
            infos[agent] = info
            
            # Track episode-level success (any agent reaching goal counts as episode success)
            is_already_successful = agent in self.success_agents
            
            if info.get('outcome') == 'success':
                self.current_episode_has_success = True
                # PERSISTENCE LOGIC:
                # If agent succeeded, DO NOT terminate. Keep it alive to hover.
                # Mark as successful
                self.success_agents.add(agent)
                # Clear termination so it stays in the loop
                term = False
                terminations[agent] = False
                
                # BUGFIX: Prevent cascading rewards
                # If agent was ALREADY successful before this step, logic in compute_reward 
                # still adds R_goal because it's inside radius. We must subtract it.
                if is_already_successful:
                    R_goal = self.config['reward'].get('R_goal', 100.0)
                    rewards[agent] -= R_goal
                    info['outcome'] = 'staying_at_goal'
            
            # Refined Persistence:
            if is_already_successful:
                term = False # Never terminate successful agents
            
            # Update prev state to include previous action for smoothness penalty
            dist = np.linalg.norm(np.array(st['pos']) - self.goals[agent])
            
            # Decrement grace period counter
            if self.collision_grace_steps[agent] > 0:
                self.collision_grace_steps[agent] -= 1
            
            # Track collision state (with grace period)
            if self.collision_grace_steps[agent] <= 0:
                # Grace period expired, track actual collisions
                self.collision_state[agent] = st['collision']
            else:
                # During grace period, ignore collisions
                self.collision_state[agent] = False
            
            self.prev_states[agent] = {
                'dist_to_goal': dist,
                'prev_action': actions_for_observation.get(agent, np.zeros(4, dtype=np.float32)),
                'was_colliding': self.collision_state[agent]  # Track for next step
            }

        # Logging
        if self.logger:
            # Flatten data for log
            log_data = {}
            for agent in self.agents:
                st = current_states[agent]
                d = {
                    'px': st['pos'][0], 'py': st['pos'][1], 'pz': st['pos'][2],
                    'vx': st['vel'][0], 'vy': st['vel'][1], 'vz': st['vel'][2],
                    'yaw': st['yaw'],
                    'gx': self.goals[agent][0], 'gy': self.goals[agent][1], 'gz': self.goals[agent][2],
                    'reward': rewards[agent],
                    'cumulative_reward': self.cumulative_rewards.get(agent, 0.0),
                    'collision': st['collision'],
                    'collision_name': infos[agent].get('collision_name'),
                    'sep_min': infos[agent].get('sep_min'),
                    'min_obs_dist': infos[agent].get('min_obs_dist'),
                    'r_obs_proximity': infos[agent].get('r_obs_proximity'),
                    'goal_reached': infos[agent].get('outcome') == 'success',
                    'outcome': infos[agent].get('outcome', 'running')
                }
                # Actions
                if agent in actions:
                    act = actions[agent]
                    # Handle both 3-DOF (simplified) and 4-DOF (original) action spaces
                    if len(act) == 3:
                        # Simplified: [speed, yaw_rate, climb_rate]
                        d['ax'] = float(act[0])  # speed
                        d['yaw_rate'] = float(act[1])
                        d['az'] = float(act[2])  # climb_rate
                    else:
                        # Original: [vx, vy, vz, yaw_rate]
                        d['ax'] = float(act[0])  # vx_body
                        d['ay'] = float(act[1])  # vy_body
                        d['az'] = float(act[2])  # vz_enu
                        d['yaw_rate'] = float(act[3])
                
                log_data[agent] = d
            
            self.logger.log_step(self.step_count, time.time(), log_data)

        # === ROBUST TIME SYNCHRONIZATION ===
        t2 = time.time() # Logic End
        
        # Ensure exactly 'self.dt' simulation seconds pass before next step.
        # This decouples physics accuracy from hardware performance/clock speed.
        if not self.smoke_test:
            target_duration_nanos = int(self.dt * 1e9)
            
            # If this is the first real step or we lost track, reset baseline
            if self.last_sim_time_nanos == 0:
                self.last_sim_time_nanos = self.airsim_client.get_sim_time()
            
            target_sim_time = self.last_sim_time_nanos + target_duration_nanos
            
            # Wait loop
            current_sim_time = self.airsim_client.get_sim_time()
            wait_start_wall = time.time()
            
            # Timeout safety: Don't freeze forever if AirSim crashes (e.g. 10s wall timeout)
            while current_sim_time < target_sim_time:
                # Sleep briefly to avoid hammering RPC
                # Adaptive sleep: if far behind, sleep longer
                delta_ns = target_sim_time - current_sim_time
                if delta_ns > 1e8: # > 0.1s sim time left
                     time.sleep(0.01) # Sleep 10ms wall
                elif delta_ns > 1e7: # > 0.01s sim time left
                     time.sleep(0.001) # Sleep 1ms wall
                else:
                    pass # Busy wait for final precision
                
                check_time = self.airsim_client.get_sim_time()
                # Handle potential reset/clock wrap (unlikely but safe)
                if check_time < current_sim_time:
                     logger.warning("Simulation time reset detected!")
                     target_sim_time = check_time # Exit loop
                current_sim_time = check_time
                
                if time.time() - wait_start_wall > 20.0:
                    logger.warning("Time sync timed out (AirSim frozen?)")
                    break
            
            self.last_sim_time_nanos = current_sim_time
            
            self.last_sim_time_nanos = current_sim_time
            
            # Log real-time speed factor every 10 steps
            actual_wall_duration = time.time() - t0
            # Performance logging removed requested by user

        # Remove dead agents
        self.agents = [
            a for a in self.agents 
            if not terminations[a] and not truncations[a]
        ]
        
        if not self.agents and self.logger:
            self.logger.end_episode()

        return observations, rewards, terminations, truncations, infos

    def _calculate_priority(self, agent_id, state):
        """Calculate local priority score for CD&R."""
        dist = np.linalg.norm(np.array(state['pos']) - self.goals[agent_id])
        speed = np.linalg.norm(state['vel'])
        
        # Priority: Closer to goal + Higher speed = Higher priority
        score = (100.0 - min(dist, 100.0)) / 100.0 + (min(speed, 10.0) / 10.0)
        
        # Tie-breaker: using agent number
        try:
            agent_num = int(agent_id.replace(self.agent_name_prefix, ""))
            score += agent_num * 0.001
        except:
            pass
        return float(score)

    def _apply_cooperative_arbitration(self, actions_to_send, states):
        """
        Apply rule-based cooperative arbitration to stabilize RL behavior.
        Uses Priority Score and Simple TTC to identify yielding agents.
        """
        modified_actions = actions_to_send.copy()
        shield_threshold = self.coop_config.get('shield_threshold_ttc', 3.0)
        use_shield = self.coop_config.get('use_action_shield', False)
        
        # Track which agents are involved in a managed conflict for reward tracking
        for agent in self.agents:
            states[agent]['is_yielding'] = False
            states[agent]['is_maintainer'] = False

        # Analyze pairs for conflicts
        # O(N^2) but N is small (usually 3-10)
        for i, a in enumerate(self.agents):
            if a not in modified_actions: continue
            
            p_a = np.array(states[a]['pos'])
            v_a = np.array(states[a]['vel'])
            prio_a = self._calculate_priority(a, states[a])
            
            for b in self.agents[i+1:]:
                if b not in modified_actions: continue
                
                p_b = np.array(states[b]['pos'])
                v_b = np.array(states[b]['vel'])
                prio_b = self._calculate_priority(b, states[b])
                
                # Check for conflict via simple TTC
                rel_pos = p_b - p_a
                rel_vel = v_a - v_b
                dist = np.linalg.norm(rel_pos)
                
                if dist < 0.1: continue 
                
                # Closing speed > 0 means they are approaching
                closing_speed = np.dot(rel_pos, rel_vel) / dist
                if closing_speed > 0:
                    ttc = dist / closing_speed
                    if ttc < shield_threshold:
                        # Conflict detected! Determine yielder
                        if prio_a > prio_b:
                            yielder, maintainer = b, a
                        else:
                            yielder, maintainer = a, b
                        
                        states[yielder]['is_yielding'] = True
                        states[maintainer]['is_maintainer'] = True
                        
                        # Apply bias to yielder
                        y_act = modified_actions[yielder].copy()
                        
                        # 1. Longitudinal Bias (Slow down)
                        factor = 0.4 if use_shield else 0.7
                        y_act[0] *= factor # vx_body
                        
                        # 2. Lateral Bias (Steer away from collision)
                        # Perpendicular vector to rel_pos
                        if dist < 10.0:
                             # Simple 2D avoidance nudge in body frame
                             # This is a heuristic: if maintainer is to my front-right, nudge left.
                             # For simplicity, we just add a small fixed vy_body nudge
                             # depending on relative bearing.
                             rel_pos_body = self._world_to_body(rel_pos, states[yielder]['yaw'])
                             if rel_pos_body[1] > 0: # maintainer is on the right
                                 y_act[1] -= 1.5   # steer left
                             else:
                                 y_act[1] += 1.5   # steer right
                            
                        modified_actions[yielder] = y_act
                        
        return modified_actions

    def _world_to_body(self, vec_world, yaw):
        """Rotate vector from world to body frame."""
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        vx = vec_world[0] * cos_yaw - vec_world[1] * sin_yaw
        vy = vec_world[0] * sin_yaw + cos_yaw * vec_world[1]
        return np.array([vx, vy, vec_world[2]])

    def _get_obs(self, cached_states=None):
        # reuse cached_states from step()
        states = cached_states if cached_states is not None else self.airsim_client.get_drone_states()
        
        # --- COMMUNICATION MODEL SIMULATION ---
        # 1. Update ground-truth messages for all agents
        current_step_messages = {}
        for agent in self.possible_agents:
            st = states[agent]
            # Content of a "message"
            msg = {
                'pos': np.array(st['pos']),
                'vel': np.array(st['vel']),
                'goal_dir': (self.goals[agent] - st['pos']) / (np.linalg.norm(self.goals[agent] - st['pos']) + 1e-6),
                'wp_dist': np.linalg.norm(self.goals[agent] - st['pos']),
                'priority': self._calculate_priority(agent, st),
                'step': self.step_count
            }
            current_step_messages[agent] = msg
            
            # Store in history
            self.message_history[agent].append(msg)
            # Keep only needed history for max delay
            max_delay = self.comm_config.get('delay_steps', 2)
            if len(self.message_history[agent]) > max_delay + 1:
                self.message_history[agent].pop(0)

        # 2. Simulate delivery for each agent (Receiver-centric)
        delivered_neighbor_info = {agent: {} for agent in self.agents}
        drop_prob = self.comm_config.get('drop_prob', 0.1)
        delay_steps = self.comm_config.get('delay_steps', 1)
        comm_radius = self.comm_config.get('radius', 50.0)
        stale_limit = self.comm_config.get('stale_limit', 5)

        for receiver in self.agents:
            p_rec = np.array(states[receiver]['pos'])
            
            for sender in self.possible_agents:
                if sender == receiver: continue
                
                # Check Radius
                p_sen = np.array(states[sender]['pos'])
                dist = np.linalg.norm(p_rec - p_sen)
                
                if dist > comm_radius:
                    # Too far - no message delivery, use last delivered if not too stale
                    pass
                else:
                    # within radius - simulate delay and drop
                    # Get delayed message
                    hist = self.message_history[sender]
                    # if delay=1, we want hist[-2] (last step's msg). if delay=0, hist[-1].
                    idx = -(delay_steps + 1)
                    if abs(idx) <= len(hist):
                        delayed_msg = hist[idx]
                        
                        # Simulate Drop
                        if np.random.random() > drop_prob:
                            # Delivered! Update last_delivered
                            self.last_delivered_messages[receiver][sender] = delayed_msg
                
                # Check Staleness of last delivered
                last_msg = self.last_delivered_messages[receiver].get(sender)
                if last_msg:
                    age = self.step_count - last_msg['step']
                    if age <= stale_limit:
                        # Still usable
                        # Freshness: 1.0 (new) to 0.0 (stale_limit)
                        freshness = max(0.0, (stale_limit - age) / stale_limit)
                        
                        delivery_info = last_msg.copy()
                        delivery_info['freshness'] = freshness
                        delivered_neighbor_info[receiver][sender] = delivery_info

        # --- BUILDING OBSERVATIONS ---
        obs_dict = {}
        obs_config = self.config['observation']
        R_min = obs_config.get('R_min', 8.0)
        R_gain = obs_config.get('R_gain', 10.0)
        
        for agent in self.agents:
            v_agent = np.array(states[agent]['vel'])
            airspeed = np.linalg.norm(v_agent)
            R = max(R_min, R_gain * airspeed)
            p_agent = np.array(states[agent]['pos'])
            
            # Neighbors collection (Using DELIVERED INFO for drones)
            neighbors = []
            
            # 1. Add other agents (using communication model results)
            for other, d_info in delivered_neighbor_info[agent].items():
                p_o = d_info['pos']
                dist = np.linalg.norm(p_o - p_agent)
                if dist <= R:
                    # Drone neighbor with intent
                    neighbors.append({
                        'pos': d_info['pos'],
                        'vel': d_info['vel'],
                        'goal_dir': d_info['goal_dir'],
                        'wp_dist': d_info['wp_dist'],
                        'priority': d_info['priority'],
                        'freshness': d_info['freshness'],
                        'dist': dist,
                        'is_obstacle': False
                    })
            
            # 2. Add LiDAR obstacles (Ground Truth - assumed perfect sensors)
            lidar_obstacles = states[agent].get('lidar_obstacles', [])
            for obs in lidar_obstacles:
                if obs['dist'] <= R:
                    neighbors.append(obs)
            
            neighbors.sort(key=lambda x: x['dist'])
            
            # Previous action
            prev_action = self.prev_states.get(agent, {}).get('prev_action', None)
            
            # Obstacle rays
            obstacle_rays = self._get_obstacle_rays(p_agent, lidar_obstacles, num_rays=8)
            states[agent]['obstacle_rays'] = obstacle_rays
            
            # VO features
            from .velocity_obstacles import compute_velocity_obstacle_for_neighbors
            K = self.config['observation'].get('K', 5)
            vo_flags, vo_ttcs, vo_cpa = compute_velocity_obstacle_for_neighbors(
                pos_self=p_agent,
                vel_self=v_agent,
                neighbors=neighbors,
                radius=self.config['reward'].get('d_safe', 5.0) / 2.0,
                K=K 
            )
            states[agent]['vo_collision_flags'] = vo_flags.astype(np.float32)
            states[agent]['vo_ttcs'] = vo_ttcs.astype(np.float32)
            states[agent]['vo_cpa_dists'] = vo_cpa.astype(np.float32)
            
            # Waypoint features (PHASE 1)
            if agent in self.planned_paths and len(self.planned_paths[agent]) > 0:
                wp_idx = self.current_waypoint_idx.get(agent, 0)
                if wp_idx >= len(self.planned_paths[agent]):
                    wp_idx = len(self.planned_paths[agent]) - 1
                
                waypoint = np.array(self.planned_paths[agent][wp_idx])
                wp_direction = waypoint - p_agent
                wp_distance = np.linalg.norm(wp_direction)
                
                if wp_distance < 3.0 and wp_idx < len(self.planned_paths[agent]) - 1:
                    self.current_waypoint_idx[agent] = wp_idx + 1
                    waypoint = np.array(self.planned_paths[agent][self.current_waypoint_idx[agent]])
                    wp_direction = waypoint - p_agent
                    wp_distance = np.linalg.norm(wp_direction)
                
                if wp_distance > 1e-6:
                    wp_dir_norm = wp_direction / wp_distance
                else:
                    wp_dir_norm = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                
                states[agent]['waypoint_direction'] = wp_dir_norm
                states[agent]['waypoint_distance'] = wp_distance
            else:
                goal_dir = self.goals[agent] - p_agent
                goal_dist = np.linalg.norm(goal_dir)
                if goal_dist > 1e-6:
                    states[agent]['waypoint_direction'] = goal_dir / goal_dist
                else:
                    states[agent]['waypoint_direction'] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                states[agent]['waypoint_distance'] = goal_dist
            
            obs = build_observation(
                states[agent], 
                neighbors, 
                self.goals[agent], 
                self.config['observation'],
                prev_action=prev_action
            )
            obs_dict[agent] = obs
        return obs_dict

    def _get_obstacle_rays(self, pos, lidar_obstacles, num_rays=8):
        """
        Get 8-directional obstacle distances (N, NE, E, SE, S, SW, W, NW).
        
        Args:
            pos: Agent position [x, y, z]
            lidar_obstacles: List of obstacle dicts from LiDAR
            num_rays: Number of angular bins (default 8)
        
        Returns:
            distances: Array of length num_rays with minimum distance in each direction
        """
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        distances = np.full(num_rays, 50.0)  # Max range (from LiDAR config)
        
        for obs in lidar_obstacles:
            obs_pos = np.array(obs['pos'])
            dx = obs_pos[0] - pos[0]
            dy = obs_pos[1] - pos[1]
            
            # Compute angle to obstacle (horizontal plane only)
            angle = np.arctan2(dy, dx)
            
            # Find corresponding ray bin
            ray_idx = int((angle % (2*np.pi)) / (2*np.pi) * num_rays)
            
            # Update minimum distance in that direction
            distances[ray_idx] = min(distances[ray_idx], obs['dist'])
        
        return distances

