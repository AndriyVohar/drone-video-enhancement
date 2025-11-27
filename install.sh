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
