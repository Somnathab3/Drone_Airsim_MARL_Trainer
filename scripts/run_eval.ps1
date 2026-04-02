$env:PYTHONPATH = "F:\Projects\PhD\Drone_Air_Sim\universal-uav-rl"
param (
    [string]$Checkpoint
)

if (-not $Checkpoint) {
    Write-Error "Please provide a checkpoint path via -Checkpoint argument."
    exit 1
}

.\.venv\Scripts\python -m evaluation.evaluate_policy_airsim --checkpoint $Checkpoint
