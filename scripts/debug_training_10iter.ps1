# Debug Training - First 10 Iterations Only
# Saves detailed logs to analyze reward patterns

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

Write-Host "`n=== Debug Training: 10 Iterations ===" -ForegroundColor Cyan
Write-Host "This will run 10 training iterations and save detailed logs" -ForegroundColor Yellow
Write-Host "Logs location: data/episodes/" -ForegroundColor White
Write-Host ""

# Run debug training (assumes venv is already activated)
python training/train_rllib_ppo.py --debug-iterations 10

Write-Host "`n=== Debug Run Complete ===" -ForegroundColor Green
Write-Host "Check data/episodes/ for episode CSVs" -ForegroundColor Yellow
Write-Host "Analyze reward patterns, collisions, and outcomes" -ForegroundColor Yellow
