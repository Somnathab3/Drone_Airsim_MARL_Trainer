try:
    import airsim
except ImportError:
    airsim = None
import numpy as np
import time
from loguru import logger
from .frames import ned_to_enu_pos, ned_to_enu_vel, quat_to_yaw_enu

class AirSimClientWrapper:
    def __init__(self, num_agents=1, ip="127.0.0.1", port=41451):
        self.ip = ip
        self.port = port
        self.num_agents = num_agents
        self.drone_names = [f"Drone{i}" for i in range(num_agents)]
        self.connection_failures = 0
        self.max_retries = 3
        
        # Initial connection
        self._connect()
        
    def _connect(self):
        """Establish or re-establish connection to AirSim"""
        try:
            self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
            self.client.confirmConnection()
            
            # Enable API control for all drones
            for name in self.drone_names:
                self.client.enableApiControl(True, vehicle_name=name)
                self.client.armDisarm(True, vehicle_name=name)
            
            self.connection_failures = 0
            logger.info("Successfully connected to AirSim")
        except Exception as e:
            raise RuntimeError(f"Could not connect to AirSim at {self.ip}:{self.port}. Is Unreal/AirSim running? Error: {e}")
    
    def _safe_call(self, func, *args, **kwargs):
        """Wrapper for AirSim calls with automatic retry on failure"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, Exception) as e:
                self.connection_failures += 1
                logger.warning(f"AirSim call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)
                    try:
                        # Try to reconnect
                        self._connect()
                    except Exception:
                        pass
                else:
                    logger.error(f"Failed after {self.max_retries} attempts. Connection may be lost.")
                    raise
            except Exception as e:
                # For non-connection errors, fail immediately
                logger.error(f"Unexpected error in AirSim call: {e}")
                raise

    def verify_connection(self):
        """Check if connection is still alive"""
        try:
            return self.client.ping()
        except Exception:
            return False

    def get_sim_time(self):
        """Get current simulation timestamp in nanoseconds."""
        try:
            # getMultirotorState returns a state object with timestamp
            # We can use any drone's state or a generic API if available.
            # getMultirotorState is reliable.
            state = self.client.getMultirotorState(vehicle_name=self.drone_names[0])
            return state.timestamp
        except Exception as e:
            logger.warning(f"Failed to get sim time: {e}")
            return 0

    def reset(self, spawn_positions=None):
        """
        Reset AirSim and optionally teleport drones to spawn positions.
        """
        def _do_reset():
            self.client.reset()
            time.sleep(0.2)
            
            for name in self.drone_names:
                self.client.enableApiControl(True, vehicle_name=name)
                self.client.armDisarm(True, vehicle_name=name)
            
            if spawn_positions:
                for name in self.drone_names:
                    if name in spawn_positions:
                        x_enu, y_enu, z_enu = spawn_positions[name]
                        x_ned = y_enu
                        y_ned = x_enu
                        z_ned = -z_enu
                        pose = airsim.Pose(airsim.Vector3r(x_ned, y_ned, z_ned), airsim.Quaternionr(0, 0, 0, 1))
                        self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=name)
                time.sleep(0.2)
                for name in self.drone_names:
                    self.client.moveByVelocityBodyFrameAsync(
                        0, 0, 0, duration=0.5,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                        vehicle_name=name
                    ).join()
            else:
                for name in self.drone_names:
                    self.client.takeoffAsync(vehicle_name=name).join()
            time.sleep(0.5)
        
        try:
            self._safe_call(_do_reset)
        except Exception as e:
            logger.error(f"Failed to reset AirSim after retries: {e}")

    def get_obstacles(self):
        """
        Query AirSim for static obstacles (Blocks, Walls, Cylinders etc).
        Returns a list of dicts: {'name': str, 'pos': np.array(3), 'scale': np.array(3)}
        Position is in ENU.
        """
        obstacles = []
        try:
            # Regex to find likely obstacles based on environment naming conventions
            regex = "TemplateCube.*|Wall.*|Cylinder.*|Cone.*|OrangeBall.*"
            obj_names = self.client.simListSceneObjects(name_regex=regex)
            
            for name in obj_names:
                pose = self.client.simGetObjectPose(name)
                scale = self.client.simGetObjectScale(name)
                
                # Convert position to ENU
                p_ned = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
                p_enu = ned_to_enu_pos(p_ned)
                
                # Scale is vector (x, y, z) multipliers
                # In NED, X is Forward, Y is Right.
                # In ENU, X is Right (NED Y), Y is Forward (NED X).
                # So we swap Scale X and Y.
                s_vec = np.array([scale.y_val, scale.x_val, scale.z_val])
                
                obstacles.append({
                    'name': name,
                    'pos': p_enu,
                    'scale': s_vec
                })
                
            logger.info(f"Detected {len(obstacles)} static obstacles in environment.")
            return obstacles
            
        except Exception as e:
            logger.error(f"Failed to query obstacles: {e}")
            return []

    def get_nearest_obstacles(self, vehicle_name, drone_pos_enu, num_obstacles=5, min_altitude=1.0, all_drone_positions=None):
        """Get nearest STATIC obstacle points from LiDAR data."""
        try:
            lidar_data = self._safe_call(self.client.getLidarData, lidar_name="LidarSensor1", vehicle_name=vehicle_name)
            if len(lidar_data.point_cloud) < 3: return []
            
            points_local = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            if len(points_local) == 0: return []
            
            drone_state = self._safe_call(self.client.getMultirotorState, vehicle_name=vehicle_name)
            pos_ned = drone_state.kinematics_estimated.position
            
            obstacles = []
            drone_x_enu, drone_y_enu, drone_z_enu = drone_pos_enu
            
            for pt_local in points_local:
                x_local, y_local, z_local = pt_local
                x_world_ned = pos_ned.x_val + x_local
                y_world_ned = pos_ned.y_val + y_local
                z_world_ned = pos_ned.z_val + z_local
                
                x_enu = y_world_ned
                y_enu = x_world_ned
                z_enu = -z_world_ned
                
                if drone_z_enu > min_altitude and z_enu < 1.0: continue
                
                dx = x_enu - drone_x_enu
                dy = y_enu - drone_y_enu
                dz = z_enu - drone_z_enu
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist < 0.5: continue
                
                is_drone = False
                if all_drone_positions is not None:
                    for drone_pos in all_drone_positions:
                        if np.sqrt((x_enu - drone_pos[0])**2 + (y_enu - drone_pos[1])**2 + (z_enu - drone_pos[2])**2) < 2.0:
                            is_drone = True
                            break
                if not is_drone:
                    obstacles.append({'pos': [x_enu, y_enu, z_enu], 'vel': [0.0, 0.0, 0.0], 'dist': dist, 'is_obstacle': True})
            
            obstacles.sort(key=lambda x: x['dist'])
            return obstacles[:num_obstacles]
        except Exception as e:
            logger.warning(f"Failed to get LiDAR data for {vehicle_name}: {e}")
            return []

    def get_drone_states(self):
        """Get state of all drones in ENU."""
        states = {}
        for name in self.drone_names:
            try:
                response = self._safe_call(self.client.getMultirotorState, vehicle_name=name)
                pos_ned = response.kinematics_estimated.position
                p_enu = ned_to_enu_pos([pos_ned.x_val, pos_ned.y_val, pos_ned.z_val])
                
                vel_ned = response.kinematics_estimated.linear_velocity
                v_enu = ned_to_enu_vel([vel_ned.x_val, vel_ned.y_val, vel_ned.z_val])
                
                orient = response.kinematics_estimated.orientation
                yaw_enu = quat_to_yaw_enu(orient.x_val, orient.y_val, orient.z_val, orient.w_val)
                
                collision_info = self._safe_call(self.client.simGetCollisionInfo, vehicle_name=name)
                states[name] = {
                    "pos": p_enu, 
                    "vel": v_enu, 
                    "yaw": yaw_enu, 
                    "collision": collision_info.has_collided,
                    "collision_name": collision_info.object_name if collision_info.has_collided else None
                }
            except Exception as e:
                logger.warning(f"Failed to get state for {name}: {e}. Using last known state.")
                states[name] = {"pos": [0.0, 0.0, 5.0], "vel": [0.0, 0.0, 0.0], "yaw": 0.0, "collision": False}
        return states

    def apply_actions(self, actions, dt, action_frequency=1):
        """
        Send velocity commands to drones using World Frame conversion and Yaw Angle integration.
        
        FIXED LOGIC (Option A):
        - Use `DrivetrainType.ForwardOnly` so drone nose follows velocity vector.
        - Use `YawMode(is_rate=False)` with a target angle to avoid conflict with ForwardOnly.
        - Manually integrate yaw_rate into target yaw angle.
        """
        futures = []
        for name, action in actions.items():
            try:
                vx_body, vy_body, vz_enu, yaw_rate = action
                
                # 1. Get current drone orientation (Yaw)
                state = self._safe_call(self.client.getMultirotorState, vehicle_name=name)
                orientation = state.kinematics_estimated.orientation
                _, _, yaw = airsim.to_eularian_angles(orientation)
                
                # 2. Convert Body Frame (Forward/Left/Up) to AirSim Body (Forward/Right/Down)
                as_vx_body = float(vx_body)
                as_vy_body = -float(vy_body) 
                
                # 3. Rotate to World NED Velocity
                c = np.cos(yaw)
                s = np.sin(yaw)
                vx_ned = as_vx_body * c - as_vy_body * s
                vy_ned = as_vx_body * s + as_vy_body * c
                vz_ned = -float(vz_enu)
                
                # 4. Calculate duration and Target Yaw
                command_duration = dt * action_frequency
                target_yaw_rad = yaw + float(yaw_rate) * command_duration
                target_yaw_deg = float(np.degrees(target_yaw_rad))
                
                # 5. Yaw Mode: Use Angle (False) to match ForwardOnly
                # If we used is_rate=True, we CANNOT use ForwardOnly.
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=target_yaw_deg)
                
                # 6. Send command
                f = self._safe_call(
                    self.client.moveByVelocityAsync,
                    vx_ned, vy_ned, vz_ned,
                    duration=command_duration,
                    drivetrain=airsim.DrivetrainType.ForwardOnly, 
                    yaw_mode=yaw_mode,
                    vehicle_name=name
                )
                futures.append(f)
            except Exception as e:
                logger.warning(f"Failed to apply action for {name}: {e}")
                class DummyFuture:
                    def join(self): pass
                futures.append(DummyFuture())
            
        return futures
        return futures
