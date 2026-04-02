
import airsim
import time
import math
import argparse

def probe_bounds(step_size=10, max_dist=150, altitude=-10):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print(f"Taking off to {altitude}m...")
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, altitude, 5).join()
    
    directions = [
        ("North (+X)", 1, 0),
        ("South (-X)", -1, 0),
        ("East (+Y)", 0, 1),
        ("West (-Y)", 0, -1)
    ]
    
    results = {}

    for name, dx, dy in directions:
        print(f"\nProbings {name}...")
        
        # Reset to center for each direction safely
        client.moveToPositionAsync(0, 0, altitude, 10).join()
        
        current_dist = 0
        safe_dist = 0
        
        while current_dist < max_dist:
            target_dist = current_dist + step_size
            tx = dx * target_dist
            ty = dy * target_dist
            
            print(f"  Moving to {target_dist}m ({tx:.1f}, {ty:.1f})...")
            
            # Move
            client.moveToPositionAsync(tx, ty, altitude, 5).join()
            
            # Check collision
            collision_info = client.simGetCollisionInfo()
            if collision_info.has_collided:
                print(f"  [COLLISION] Hit object: {collision_info.object_name} at {target_dist}m")
                break
            
            # Check if we are still effectively moving (basic check)
            pos = client.getMultirotorState().kinematics_estimated.position
            actual_dist = math.sqrt(pos.x_val**2 + pos.y_val**2)
            
            # Verify we actually reached roughly the target
            if abs(actual_dist - target_dist) > 5.0:
                 print(f"  [WARNING] Stopped moving? Target: {target_dist}, Actual: {actual_dist}")
                 # Could be edge of world invisible wall?
                 break
            
            safe_dist = current_dist
            current_dist = target_dist
            time.sleep(0.5)
            
        results[name] = safe_dist
        print(f"  Max Safe Distance: {safe_dist}m")

    print("\n=== BOUNDARY PROBE RESULTS ===")
    for direction, dist in results.items():
        print(f"{direction}: {dist}m")
        
    client.reset()
    client.enableApiControl(False)

if __name__ == "__main__":
    probe_bounds()
