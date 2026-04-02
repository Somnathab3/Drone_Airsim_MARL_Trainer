# Create venv
if (!(Test-Path .venv)) {
    Write-Host "Creating virtual environment..."
    py -3.11 -m venv .venv
} else {
    Write-Host "Virtual environment already exists."
}

# Activate instructions
Write-Host "To activate, run: .\.venv\Scripts\Activate.ps1"

# Upgrade pip
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
