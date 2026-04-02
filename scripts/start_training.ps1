# Start Full Training - 10 Agents, 1M Timesteps, GPU

# Change to project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

# Disable Ray metrics exporter warnings
$env:RAY_DISABLE_DASHBOARD = "1"
$env:RAY_DISABLE_USAGE_STATS = "1"
$env:RAY_DEDUP_LOGS = "0"

# Add project root to PYTHONPATH
$env:PYTHONPATH = $projectRoot

# Check GPU availability
Write-Host "`n=== Checking GPU Availability ===" -ForegroundColor Cyan
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

Write-Host "`n=== Starting Training ===" -ForegroundColor Green
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Agents: 10" -ForegroundColor White
Write-Host "  - Target Timesteps: 1,000,000" -ForegroundColor White
Write-Host "  - GPU: Enabled" -ForegroundColor White
Write-Host "  - Checkpoints: Every 50 iterations" -ForegroundColor White
Write-Host "  - Checkpoint Dir: models/checkpoints" -ForegroundColor White
Write-Host ""

# Start training
python training/train_rllib_ppo.py

Write-Host "`n=== Training Complete ===" -ForegroundColor Green
