# Navigate to project root
Set-Location F:\Projects\PhD\Drone_Air_Sim\universal-uav-rl

# Set Python path
$env:PYTHONPATH = "F:\Projects\PhD\Drone_Air_Sim\universal-uav-rl"

# Run training
& .\.venv\Scripts\python.exe training\train_rllib_ppo.py
