# Ensure pip is up to date
python -m pip install --upgrade pip setuptools wheel

# Install numpy first to avoid AirSim build errors
python -m pip install numpy

# Install remaining requirements
python -m pip install -r requirements.txt
