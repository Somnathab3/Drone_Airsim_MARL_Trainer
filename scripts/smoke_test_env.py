try:
    from envs.airsim_client import AirSimClientWrapper
    print("Importing AirSimClientWrapper successful.")
except ImportError as e:
    print(f"ImportError: {e}")
    exit(1)

import time
import numpy as np

def main():
    print("Starting Smoke Test...")
    # Mock config or connect to real if valid
    # This script assumes AirSim SHOULD be running, but we handle connection failure gracefully in wrapper.
    
    wrapper = AirSimClientWrapper(num_agents=1)
    
    if wrapper.client.ping():
        print("Connected to AirSim!")
    else:
        print("Could not connect to AirSim. Make sure it is running.")
        # We can't really fail here if user hasn't started it, but let's assume they might have.
        # But for 'smoke test' of code logic, we might want a mock mode.
        pass

    try:
        print("Resetting...")
        wrapper.reset()
        print("Reset successful.")
        
        print("Getting state...")
        states = wrapper.get_drone_states()
        print(f"States: {states}")
        
        print("Applying action (hover)...")
        # vx=0, vy=0, vz=0, yaw_rate=0
        actions = {"Drone0": [0, 0, 0, 0]} 
        wrapper.apply_actions(actions, dt=0.1)
        time.sleep(0.1)
        
        print("Smoke Test Complete. Code seems importable and runnable.")
        
    except Exception as e:
        print(f"Smoke Test Failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
