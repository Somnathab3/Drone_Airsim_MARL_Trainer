import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from envs import frames

def test_ned_to_enu_pos():
    ned = np.array([1.0, 2.0, 3.0])
    enu = frames.ned_to_enu_pos(ned)
    # x_e = y_n_ed = 2.0
    # y_n = x_n_ed = 1.0
    # z_u = -z_d = -3.0
    # Expected: [2.0, 1.0, -3.0]
    np.testing.assert_allclose(enu, [2.0, 1.0, -3.0])

    # Batch
    ned_batch = np.array([[1, 2, 3], [4, 5, 6]])
    enu_batch = frames.ned_to_enu_pos(ned_batch)
    expected = np.array([[2, 1, -3], [5, 4, -6]])
    np.testing.assert_allclose(enu_batch, expected)

def test_quat_to_yaw_enu():
    # Identity quaternion in NED (North facing)
    # Body X (Forward) -> North (1, 0, 0) NED
    # ENU: North is Y axis. So Yaw should be 90 deg (pi/2).
    q_ident = [0, 0, 0, 1]
    yaw = frames.quat_to_yaw_enu(*q_ident)
    np.testing.assert_allclose(yaw, np.pi/2)

    # 90 deg Yaw (East facing in NED)
    # Rotation about Z-axis by 90 deg.
    # q = [0, 0, sin(45), cos(45)]
    r = Rotation.from_euler('z', 90, degrees=True)
    q_east = r.as_quat()
    yaw_east = frames.quat_to_yaw_enu(*q_east)
    # East in ENU is X axis. Yaw should be 0.
    np.testing.assert_allclose(yaw_east, 0.0, atol=1e-6)

def test_world_to_body_2d():
    # Case 1: Yaw=0 (East), Target=East (1,0)
    # Should be (1,0) in body
    yaw = 0.0
    vec_world = [1.0, 0.0]
    vec_body = frames.world_to_body_2d(vec_world, yaw)
    np.testing.assert_allclose(vec_body, [1.0, 0.0], atol=1e-6)

    # Case 2: Yaw=90 (North), Target=North (0,1)
    # Should be (1,0) in body
    yaw = np.pi/2
    vec_world = [0.0, 1.0]
    vec_body = frames.world_to_body_2d(vec_world, yaw)
    np.testing.assert_allclose(vec_body, [1.0, 0.0], atol=1e-6)

    # Case 3: Yaw=90 (North), Target=East (1,0)
    # Body is facing North. Target is East (Right).
    # Body frame Y is Left. So Target should be -Y (Right).
    # Expect Body: (0, -1)
    yaw = np.pi/2
    vec_world = [1.0, 0.0]
    vec_body = frames.world_to_body_2d(vec_world, yaw)
    np.testing.assert_allclose(vec_body, [0.0, -1.0], atol=1e-6)
