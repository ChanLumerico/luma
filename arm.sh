# Download Miniforge
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

# Run the installer
bash Miniforge3-MacOSX-arm64.sh

# Initialize Miniforge
source ~/miniforge3/bin/activate

# Optionally, add to your shell startup file
echo "source ~/miniforge3/bin/activate" >> ~/.zshrc  # or ~/.bashrc
source ~/.zshrc  # or source ~/.bashrc

# Create and activate a new environment
conda create -n luma-arm-env python=3.12
conda activate luma-arm-env

# Verify the platform
python -c "import platform; print(platform.machine())"