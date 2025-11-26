#!/bin/bash

# Installation and Setup Script for Drone Video Enhancement

echo "=========================================="
echo "Drone Video Enhancement - Installation"
echo "=========================================="
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python installation..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Error: Python 3 not found. Please install Python 3.7 or higher."
    exit 1
fi

echo "✓ Python found"
echo ""

# Step 2: Create virtual environment (optional but recommended)
echo "Step 2: Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Step 3: Activate virtual environment
echo "Step 3: Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Step 4: Upgrade pip
echo "Step 4: Upgrading pip..."
pip install --upgrade pip
echo ""

# Step 5: Install dependencies
echo "Step 5: Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error installing dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Step 6: Create necessary directories
echo "Step 6: Creating directories..."
mkdir -p input output demo_output quickstart_output
echo "✓ Directories created"
echo ""

# Step 7: Run quickstart test
echo "Step 7: Running quickstart test..."
python quickstart.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Installation successful!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Place your drone video in: input/drone_video.mp4"
    echo "  2. Edit config.py to adjust parameters"
    echo "  3. Run: python main.py"
    echo ""
    echo "Or try the demo:"
    echo "  python demo.py"
    echo ""
    echo "Check quickstart_output/ for test results"
    echo "=========================================="
else
    echo "❌ Quickstart test failed. Check for errors above."
    exit 1
fi

