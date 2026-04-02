# Deploy AirSim Settings
$Source = "config\airsim_settings.json"
$DestDir = "$env:USERPROFILE\Documents\AirSim"
$DestFile = "$DestDir\settings.json"

if (-not (Test-Path $Source)) {
    Write-Error "Source file '$Source' not found!"
    exit 1
}

if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    Write-Host "Created directory: $DestDir" -ForegroundColor Cyan
}

Copy-Item -Path $Source -Destination $DestFile -Force
Write-Host "Successfully deployed settings.json to '$DestFile'" -ForegroundColor Green
Write-Host "`nNew features in this update:" -ForegroundColor Cyan
Write-Host "  • LiDAR sensors added to all drones (50m range, 16 channels)" -ForegroundColor White
Write-Host "  • Obstacle detection enabled (buildings, walls, ground, ceiling)" -ForegroundColor White
Write-Host "  • K=5 nearest entities now include both drones AND obstacles" -ForegroundColor White
Write-Host "`nIMPORTANT: You must RESTART AirSim/Unreal for these settings to take effect!" -ForegroundColor Yellow
