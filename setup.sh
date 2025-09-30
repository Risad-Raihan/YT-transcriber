#!/bin/bash
# Quick setup script for Bengali Educational Video Processing Pipeline

set -e  # Exit on error

echo "================================================"
echo "Setup Script for Bengali Video Pipeline"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: This script is designed for Linux. You may need to install dependencies manually.${NC}"
fi

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓ Python $python_version (OK)${NC}"
else
    echo -e "${RED}✗ Python $python_version is too old. Requires Python 3.10+${NC}"
    exit 1
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Step 3: Activate virtual environment
echo ""
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Step 4: Upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Step 5: Install Python dependencies
echo ""
echo "Step 5: Installing Python dependencies (this may take a while)..."
echo "   This will install PyTorch, Transformers, and other packages..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Step 6: Install system dependencies
echo ""
echo "Step 6: Installing system dependencies..."
echo "   You may be prompted for your password..."

if command -v apt-get &> /dev/null; then
    sudo apt-get update > /dev/null 2>&1
    sudo apt-get install -y ffmpeg espeak libespeak-dev
    echo -e "${GREEN}✓ System dependencies installed (ffmpeg, espeak)${NC}"
elif command -v brew &> /dev/null; then
    brew install ffmpeg espeak
    echo -e "${GREEN}✓ System dependencies installed (ffmpeg, espeak)${NC}"
else
    echo -e "${YELLOW}⚠ Could not detect package manager. Please install ffmpeg and espeak manually.${NC}"
fi

# Step 7: Create .env file
echo ""
echo "Step 7: Setting up environment variables..."
if [ -f ".env" ]; then
    echo -e "${YELLOW}.env file already exists. Skipping...${NC}"
else
    if [ -f "env.example" ]; then
        cp env.example .env
        echo -e "${GREEN}✓ Created .env file from template${NC}"
        echo -e "${YELLOW}⚠ Please edit .env and add your API keys!${NC}"
    else
        echo -e "${RED}✗ env.example not found${NC}"
    fi
fi

# Step 8: Create directories
echo ""
echo "Step 8: Creating directory structure..."
mkdir -p data/videos data/audio data/frames output logs
echo -e "${GREEN}✓ Directories created${NC}"

# Step 9: Run setup test
echo ""
echo "Step 9: Running setup verification..."
python test_setup.py

echo ""
echo "================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API keys:"
echo "     nano .env"
echo ""
echo "  2. (Optional) Edit config.json to customize settings:"
echo "     nano config.json"
echo ""
echo "  3. Run the pipeline:"
echo "     python main.py 'https://youtu.be/Qp15iVGv2oA'"
echo ""
echo "For more information, see:"
echo "  - QUICKSTART.md for quick start guide"
echo "  - README.md for full documentation"
echo ""

