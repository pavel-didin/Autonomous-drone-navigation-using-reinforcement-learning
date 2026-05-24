#!/bin/bash
# Build and install the modified gym-pybullet-drones package locally
# Requires Poetry (https://python-poetry.org/) to be installed.

echo "Uninstalling old gym_pybullet_drones (if any)..."
echo "Y" | pip uninstall gym_pybullet_drones

echo "Removing old dist/ folder..."
rm -rf dist/

echo "Building the package with Poetry..."
poetry build

echo "Installing the newly built wheel..."
pip install dist/gym_pybullet_drones-1.0.0-py3-none-any.whl

echo "Running basic test to verify installation..."
cd tests
python test_build.py
rm -rf results
cd ..

echo "Done. The custom gym-pybullet-drones environment is ready for use."
