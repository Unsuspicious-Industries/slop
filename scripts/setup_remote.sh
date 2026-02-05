#!/bin/bash
# Quick setup script for remote servers
# Usage: ./scripts/setup_remote.sh

set -e

echo "Setting up SLOP on remote server"
echo "===================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/generated/trajectories
mkdir -p data/generated/embeddings
mkdir -p data/generated/images
mkdir -p data/historical/embeddings
mkdir -p outputs/figures
mkdir -p outputs/metrics
mkdir -p outputs/reports

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Device count: {torch.cuda.device_count()}")
else:
    print("Warning: No GPU available, will use CPU")
EOF

# Test imports
echo ""
echo "Testing imports..."
python3 << EOF
try:
    import torch
    import numpy as np
    import transformers
    import diffusers
    print("Core packages imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)
EOF

echo ""
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  python scripts/run_full_analysis.py --help"
