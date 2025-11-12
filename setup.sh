#!/bin/bash
# Meeting Agent Setup Script

set -e

echo "========================================"
echo "  Meeting Agent Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA support..."
echo "If you don't have CUDA or want CPU-only, press Ctrl+C and install manually"
sleep 3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/transcripts data/recordings logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env and add your API keys:"
    echo "  nano .env"
fi

# Test GPU availability
echo ""
echo "Testing GPU availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Edit .env file and add your DeepSeek API key:"
echo "   nano .env"
echo ""
echo "3. Run the application:"
echo "   python main.py"
echo ""
echo "For more information, see README.md"
