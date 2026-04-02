import numpy as np
from scipy.spatial.transform import Rotation

def ned_to_enu_pos(pos_ned):
    """
    Convert position from NED (North, East, Down) to ENU (East, North, Up).
    NED: (x_n, y_e, z_d)
    ENU: (x_e, y_n, z_u) -> (y_e, x_n, -z_d)
    Args:
        pos_ned: array-like of shape (3,) or (N, 3)
    Returns:
        pos_enu: np.ndarray of same shape
    """
    pos_ned = np.array(pos_ned)
    if pos_ned.ndim == 1:
        return np.array([pos_ned[1], pos_ned[0], -pos_ned[2]])
    else:
        # Swap col 0 and 1, negate col 2
        return np.stack([pos_ned[:, 1], pos_ned[:, 0], -pos_ned[:, 2]], axis=-1)

def ned_to_enu_vel(vel_ned):
    """
    Convert velocity from NED to ENU. Valid for linear velocities.
    Same transformation as position.
    """
    return ned_to_enu_pos(vel_ned)

def enu_to_ned_pos(pos_enu):
    """
    Convert position from ENU to NED.
    ENU: (x_e, y_n, z_u)
    NED: (y_n, x_e, -z_u)
    Args:
        pos_enu: array-like of shape (3,) or (N, 3)
    Returns:
        pos_ned: np.ndarray of same shape
    """
    pos_enu = np.array(pos_enu)
    if pos_enu.ndim == 1:
        return np.array([pos_enu[1], pos_enu[0], -pos_enu[2]])
    else:
        return np.stack([pos_enu[:, 1], pos_enu[:, 0], -pos_enu[:, 2]], axis=-1)

def quat_to_yaw_enu(qx, qy, qz, qw):
    """
    Extract Yaw from Quaternion in ENU frame.
    Assumes quaternion rotates vectors from Body (Forward-Right-Down) to NED.
    
    Method:
    1. Rotate Body-Forward vector (1, 0, 0) by quat to get Forward in NED.
    2. Convert Forward-NED to Forward-ENU.
    3. Calculate atan2(y, x) of Forward-ENU.
    """
    # Create rotation object
    rot = Rotation.from_quat([qx, qy, qz, qw])
    
    # Body forward vector in Body frame (usually X-axis)
    # AirSim Body is Front-Right-Down.
    v_body = np.array([1, 0, 0])
    
    # Rotate to NED
    v_ned = rot.apply(v_body)
    
    # Convert to ENU
    # vs NED: (x, y, z) -> ENU: (y, x, -z)
    v_enu_x = v_ned[1]
    v_enu_y = v_ned[0]
    
    # Yaw is angle in ENU XY plane
    return np.arctan2(v_enu_y, v_enu_x)

def world_to_body_2d(vec_world_enu, yaw_enu):
    """
    Rotate a 2D vector (or 3D project to 2D) from World ENU to Body Frame (Yaw aligned).
    Args:
        vec_world_enu: (dx, dy) or (dx, dy, dz)
        yaw_enu: scalar in radians (0 = East, +CCW)
    Returns:
        (vx_body, vy_body)
    """
    vx, vy = vec_world_enu[0], vec_world_enu[1]
    # Rotate by -yaw to align with body forward
    # Forward in ENU body is usually X if we define 0 yaw as East.
    # If yaw=0 (East), and vec=(1,0) (East), body=(1,0).
    # If yaw=pi/2 (North), and vec=(0,1) (North), body=(1,0).
    c = np.cos(-yaw_enu)
    s = np.sin(-yaw_enu)
    
    dataset_x = vx * c - vy * s
    dataset_y = vx * s + vy * c
    return np.array([dataset_x, dataset_y])

def world_to_body_3d(vec_world_enu, yaw_enu):
    """
    Rotate a 3D vector from World ENU to Body Frame (Yaw aligned only).
    Does NOT compensate for pitch/roll (assuming quadrotor stabilization).
    """
    v2d = world_to_body_2d(vec_world_enu, yaw_enu)
    return np.array([v2d[0], v2d[1], vec_world_enu[2]])
