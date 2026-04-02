# AirSim Camera Setup for Training

## Overview
The updated `airsim_settings.json` now includes camera configuration to view drones from a distance during training.

## Key Settings

### 1. View Mode
```json
"ViewMode": "SpringArmChase"
```
- Follows Drone0 from behind with a spring arm
- Smooth camera movement
- Good for watching drone behavior

### 2. Camera Director
```json
"CameraDirector": {
  "X": 0, "Y": 0, "Z": -20,
  "FollowDistance": 50,
  "AttachedToVehicle": "Drone0"
}
```
- Camera stays **50 meters** behind Drone0
- Height offset: **-20 meters** (elevated view)
- Automatically tracks Drone0's movement

## How to Use

### 1. Copy Settings to AirSim
```powershell
# Windows default location:
cp config/airsim_settings.json "$env:USERPROFILE\Documents\AirSim\settings.json"

# Or manually copy to:
# C:\Users\YourName\Documents\AirSim\settings.json
```

### 2. Restart Unreal Engine
- Close Unreal/AirSim if running
- Reopen the environment
- Camera will now follow from a distance

### 3. Manual Camera Control (While Training)
Press these keys in Unreal window:

- **`** (Backtick): Toggle camera mode
- **Arrow Keys**: Pan camera
- **PageUp/PageDown**: Move camera up/down
- **F1**: Switch to vehicle 0 (Drone0)
- **F2**: Switch to vehicle 1 (Drone1)
- **0-9**: Quick switch between drones
- **\\**: Toggle recording mode
- **Backslash**: Free camera (detach from drone)

### 4. Best View Modes

**SpringArmChase** (Current):
- Follows drone smoothly
- Good for single drone focus
- Can see nearby drones

**Manual** (Press Backslash):
- Free-fly camera
- Full control with mouse + WASD
- Best for observing all 5 drones at once

**NoDisplay**:
- Headless mode (no rendering)
- Fastest for pure training
- Set `"ViewMode": "NoDisplay"` for speed

## Recommended Workflow

### During Development/Debugging
```json
"ViewMode": "SpringArmChase"
"FollowDistance": 50
```
**Why:** See what drones are doing, debug behavior

### During Full Training (Speed)
```json
"ViewMode": "NoDisplay"
```
**Why:** No rendering = 2-3x faster training

### For Recording Videos
```json
"ViewMode": "Manual"
```
Then use free camera to position for best shot

## Adjust Camera Distance

Edit `FollowDistance` in settings.json:

| Distance | Use Case |
|----------|----------|
| 20-30m | Close follow, see details |
| **50m** | **Good balance (current)** |
| 80-100m | See full swarm, distant view |
| 150m+ | Overview of large area |

## Quick Toggle Script

Create `scripts/toggle_camera.ps1`:
```powershell
# Quick toggle between training modes

param([switch]$visual, [switch]$headless)

$settingsPath = "$env:USERPROFILE\Documents\AirSim\settings.json"

if ($visual) {
    # Enable visual mode
    $settings = Get-Content config/airsim_settings.json | ConvertFrom-Json
    $settings.ViewMode = "SpringArmChase"
    $settings | ConvertTo-Json -Depth 10 | Set-Content $settingsPath
    Write-Host "Visual mode enabled - SpringArmChase at 50m"
}
elseif ($headless) {
    # Enable headless mode
    $settings = Get-Content config/airsim_settings.json | ConvertFrom-Json
    $settings.ViewMode = "NoDisplay"
    $settings | ConvertTo-Json -Depth 10 | Set-Content $settingsPath
    Write-Host "Headless mode enabled - No rendering"
}

Write-Host "Restart Unreal Engine for changes to take effect"
```

Usage:
```powershell
.\scripts\toggle_camera.ps1 -visual    # Enable visual mode
.\scripts\toggle_camera.ps1 -headless  # Disable rendering
```
