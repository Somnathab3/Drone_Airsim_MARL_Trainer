"""
PID Controller for UAV stabilization
Implements position and yaw control with PID feedback
"""
import numpy as np
from typing import Tuple, Optional


class PIDController:
    """PID controller for a single control axis"""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_max: float = 10.0):
        """
        Initialize PID controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_min: Minimum output value
            output_max: Maximum output value
            integral_max: Maximum integral accumulation (anti-windup)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max
        
        # State variables
        self.integral = 0.0
        self.error_prior = 0.0
        self.first_run = True
        
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.error_prior = 0.0
        self.first_run = True
        
    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller with new error
        
        Args:
            error: Current error (desired - actual)
            dt: Time step since last update
            
        Returns:
            Control output
        """
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        
        # Derivative term (avoid derivative kick on first run)
        if self.first_run:
            derivative = 0.0
            self.first_run = False
        else:
            derivative = (error - self.error_prior) / dt if dt > 0 else 0.0
        
        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Clamp output
        output = np.clip(output, self.output_min, self.output_max)
        
        # Save for next iteration
        self.error_prior = error
        
        return output


class UAVPIDController:
    """Complete UAV controller with position and yaw PID control"""
    
    def __init__(self, dt: float = 0.05):
        """
        Initialize UAV PID controller
        
        Args:
            dt: Control loop time step (default 50ms = 20Hz)
        """
        self.dt = dt
        
        # Position PIDs (for vx, vy, vz velocities)
        # Tuned for stable hover with moderate responsiveness
        self.pid_x = PIDController(kp=0.5, ki=0.1, kd=0.2, output_min=-5.0, output_max=5.0)
        self.pid_y = PIDController(kp=0.5, ki=0.1, kd=0.2, output_min=-5.0, output_max=5.0)
        self.pid_z = PIDController(kp=0.8, ki=0.15, kd=0.3, output_min=-3.0, output_max=3.0)
        
        # Yaw PID (for yaw rate)
        self.pid_yaw = PIDController(kp=1.0, ki=0.05, kd=0.1, output_min=-2.0, output_max=2.0)
        
    def reset(self):
        """Reset all PID controllers"""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()
        
    def compute_control(self, 
                       current_pos: np.ndarray,
                       target_pos: np.ndarray,
                       current_yaw: float,
                       target_yaw: float) -> Tuple[float, float, float, float]:
        """
        Compute velocity commands to reach target position and yaw
        
        Args:
            current_pos: Current position [x, y, z] in meters
            target_pos: Target position [x, y, z] in meters
            current_yaw: Current yaw angle in radians
            target_yaw: Target yaw angle in radians
            
        Returns:
            Tuple of (vx, vy, vz, yaw_rate) velocities in m/s and rad/s
        """
        # Position errors
        error_x = target_pos[0] - current_pos[0]
        error_y = target_pos[1] - current_pos[1]
        error_z = target_pos[2] - current_pos[2]
        
        # Yaw error (handle wrapping)
        error_yaw = target_yaw - current_yaw
        # Normalize to [-pi, pi]
        error_yaw = np.arctan2(np.sin(error_yaw), np.cos(error_yaw))
        
        # Compute velocity commands
        vx = self.pid_x.update(error_x, self.dt)
        vy = self.pid_y.update(error_y, self.dt)
        vz = self.pid_z.update(error_z, self.dt)
        yaw_rate = self.pid_yaw.update(error_yaw, self.dt)
        
        return vx, vy, vz, yaw_rate
    
    def compute_hover_control(self, 
                              current_pos: np.ndarray,
                              hover_pos: np.ndarray,
                              current_yaw: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Compute control to maintain hover at a fixed position
        
        Args:
            current_pos: Current position [x, y, z]
            hover_pos: Desired hover position [x, y, z]
            current_yaw: Current yaw angle (optional, defaults to 0)
            
        Returns:
            Tuple of (vx, vy, vz, yaw_rate) to maintain hover
        """
        # Hover means maintaining position and zero yaw
        return self.compute_control(current_pos, hover_pos, current_yaw, 0.0)
