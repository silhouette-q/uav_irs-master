#!/bin/bash

# SAC Environment Setup Script
echo "Setting up SAC (Soft Actor-Critic) Algorithm Environment..."

# Add local bin to PATH for user installations
export PATH=$PATH:$HOME/.local/bin

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install basic dependencies first
echo "Installing basic dependencies..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing SAC dependencies..."
python3 -m pip install -r requirements.txt

# Set repository permissions for full access
echo "Setting repository permissions..."
find . -type d -exec chmod 755 {} \;
find . -type f -exec chmod 644 {} \;

# Make scripts executable
chmod +x *.py *.sh

echo "Environment setup complete!"
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""