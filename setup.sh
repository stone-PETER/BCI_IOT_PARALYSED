#!/bin/bash

# BCI Real-time Classification System - Quick Start Script

echo "🧠 BCI Real-time Classification System"
echo "====================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Python
if command_exists python; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"
    PYTHON_CMD="python"
elif command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"
    PYTHON_CMD="python3"
else
    echo -e "${RED}✗ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check Node.js
if command_exists node; then
    NODE_VERSION=$(node --version 2>&1)
    echo -e "${GREEN}✓ Node.js found: $NODE_VERSION${NC}"
else
    echo -e "${RED}✗ Node.js not found. Please install Node.js 16+${NC}"
    exit 1
fi

# Check npm
if command_exists npm; then
    NPM_VERSION=$(npm --version 2>&1)
    echo -e "${GREEN}✓ npm found: $NPM_VERSION${NC}"
else
    echo -e "${RED}✗ npm not found. Please install npm${NC}"
    exit 1
fi

echo

# Check if model files exist
echo -e "${BLUE}Checking model files...${NC}"
MODEL_FILES=("bci_eegnet_model.h5" "bci_preprocessor.pkl" "bci_preprocessed_data.npz")
MISSING_FILES=()

for file in "${MODEL_FILES[@]}"; do
    if [ -f "CODE/$file" ]; then
        echo -e "${GREEN}✓ Found: CODE/$file${NC}"
    else
        echo -e "${YELLOW}⚠ Missing: CODE/$file${NC}"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Note: Some model files are missing. The system will work but may have limited functionality.${NC}"
fi

echo

# Backend setup
echo -e "${BLUE}Setting up backend...${NC}"
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo -e "${RED}Failed to find virtual environment activation script${NC}"
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Backend dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install backend dependencies${NC}"
    exit 1
fi

cd ..

# Frontend setup
echo -e "${BLUE}Setting up frontend...${NC}"
cd frontend

# Install npm dependencies
echo "Installing npm dependencies..."
npm install

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Frontend dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install frontend dependencies${NC}"
    exit 1
fi

cd ..

echo
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo
echo -e "${BLUE}To start the system:${NC}"
echo
echo -e "${YELLOW}1. Start the backend server:${NC}"
echo "   cd backend"
echo "   source venv/bin/activate  # or venv/Scripts/activate on Windows"
echo "   python app.py"
echo
echo -e "${YELLOW}2. In a new terminal, start the frontend:${NC}"
echo "   cd frontend"
echo "   npm start"
echo
echo -e "${YELLOW}3. Open your browser to:${NC}"
echo "   http://localhost:3000"
echo
echo -e "${BLUE}For more information, see README.md${NC}"