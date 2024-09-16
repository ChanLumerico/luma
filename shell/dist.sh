#!/bin/bash

# Stop the script if any command fails
set -e

# Update all submodules recursively to ensure they are up-to-date
echo "Updating submodules..."
git submodule update --init --recursive

# Clean up any old builds
echo "Cleaning old distribution files..."
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/

# Build the source distribution and wheel
echo "Building the package..."
python3 setup.py sdist bdist_wheel

# Verify that the built files are correct
echo "Build complete. Verifying the package contents..."
twine check dist/*

# Upload the package to PyPI using Twine
echo "Uploading the package to PyPI..."
twine upload dist/*

# Confirmation message
echo "Package has been successfully uploaded to PyPI!"
