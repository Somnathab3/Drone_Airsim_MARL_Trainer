# Observation Definition

The agent observes a **50-dimensional** vector of float32 values. All features use **egocentric body-frame coordinates** for better spatial awareness and control.

## Fixed-Size Vector Structure

`[ Goal(4), Ownship(3), Derivatives(3), Swarm(6), PrevAction(4), Neighbors(6 * 5) ]`

### 1. Goal Features (4) - Body Frame
* `goal_x_body`: Goal position X in body frame (forward/backward). Normalized by `d_ref`.
* `goal_y_body`: Goal position Y in body frame (left/right). Normalized by `d_ref`.
* `goal_z_enu`: Goal altitude in world frame (up/down). Normalized by `d_ref`.
* `goal_dist_norm`: Tanh-normalized Euclidean distance to goal.
  * `tanh(dist / d_ref)`

### 2. Ownship Velocity Features (3) - Body Frame
* `vx_body_norm`: Forward velocity in body frame. Normalized by `v_ref`.
* `vy_body_norm`: Lateral velocity in body frame. Normalized by `v_ref`.
* `vz_enu_norm`: Vertical velocity in world frame. Normalized by `v_ref`.

### 3. Derivative Features (3)
* `d_goal_dt`: Rate of change of distance to goal (progress rate).
* `d_sep_dt`: Rate of change of minimum separation (safety rate).
* `d_altitude_dt`: Rate of change of altitude (climb/descent rate).

### 4. Swarm Centroid Features (6) - Body Frame
Relative to swarm center of mass (Section 4.1: Swarm Coordination):
* `centroid_x_body`: Centroid position X in body frame. Normalized by `p_ref`.
* `centroid_y_body`: Centroid position Y in body frame. Normalized by `p_ref`.
* `centroid_z_enu`: Centroid altitude in world frame. Normalized by `p_ref`.
* `centroid_vx_body`: Centroid velocity X in body frame. Normalized by `v_ref_rel`.
* `centroid_vy_body`: Centroid velocity Y in body frame. Normalized by `v_ref_rel`.
* `centroid_vz_enu`: Centroid velocity Z in world frame. Normalized by `v_ref_rel`.

### 5. Previous Action (4)
Last action sent to AirSim (for action smoothness learning):
* `prev_vx_body`: Previous forward velocity command.
* `prev_vy_body`: Previous lateral velocity command.
* `prev_vz_enu`: Previous vertical velocity command.
* `prev_yaw_rate`: Previous yaw rate command.

### 6. Neighbor Features (30 = 6 * 5) - Body Frame
For the **K=5** nearest entities (drones + obstacles) within radius **R**, in egocentric body frame:

**Two Information Sources (No Duplication):**
1. **Drones**: Tracked via AirSim state queries
   - Accurate position + velocity from kinematics
   - All other agents in the swarm
   
2. **Static Obstacles**: Detected via LiDAR point cloud
   - Buildings, walls, ceiling
   - Ground (only when altitude < 1.0m)
   - **Drones filtered out** (2m threshold) to avoid duplication

**Features per entity:**
* Relative Position (3): `dx_body, dy_body, dz_enu` in body-relative coords. Normalized by `p_ref`.
* Relative Velocity (3): `dvx_body, dvy_body, dvz_enu` in body-relative coords. Normalized by `v_ref_rel`.
  * Drones: actual velocity from state
  * Static obstacles: velocity = [0, 0, 0]

**Prioritization:** All entities combined and sorted by Euclidean distance (closest first), then top K=5 selected.

Total: **50** values.

## Key Design Principles

### Egocentric Body Frame
- All spatial features (except Z-altitude) are in **body-relative coordinates**
- Transforms world positions/velocities to agent's local frame using yaw rotation
- Benefits:
  - Policy learns direction-invariant behaviors
  - Easier to learn "turn left" vs absolute world directions
  - More sample-efficient training

### World Frame Z-Axis
- Altitude (Z) kept in **ENU world frame** (up is positive)
- Reason: Gravity acts in world frame, altitude control is absolute
- Makes vertical control more intuitive

### Obstacle Detection via LiDAR
- Each drone equipped with **LiDAR sensor** (50m range, 16 channels)
- **Purpose**: Detect static environment obstacles (buildings, walls, ceiling, ground)
- **Drone filtering**: LiDAR points within 2m of any drone position are excluded
  - Prevents duplication: same drone not counted twice
  - Drones tracked via state queries (more accurate + velocity info)
  - LiDAR used ONLY for static obstacles
- **Ground filtering:** When altitude > 1.0m, ground points excluded to reduce noise
- **Integration**: Static obstacles merged with drone neighbors, sorted by distance, top K=5 selected
- **Benefits**: 
  - Predictive collision avoidance (not just reactive penalties)
  - Environment-aware navigation around buildings
  - No duplication of information

## Normalization Constants
Defaults in `config/env.yaml`:
* `d_ref`: 100 m
* `v_ref`: 10 m/s
* `p_ref`: 100 m
* `v_ref_rel`: 20 m/s
