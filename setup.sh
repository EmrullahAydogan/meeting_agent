#!/bin/bash
# Meeting Agent - Automated Linux Setup Script
# Supports both Quick (Gemini Live) and Full (Classic Mode) installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${CYAN}"
    echo "=================================================="
    echo "  $1"
    echo "=================================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found!"
        echo "Please install Python 3.10+ first:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install python3 python3-pip python3-venv"
        exit 1
    fi

    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $python_version detected"
}

check_audio() {
    print_info "Checking audio system..."

    if command -v pulseaudio &> /dev/null; then
        print_success "PulseAudio detected"
    elif command -v pactl &> /dev/null; then
        print_success "PulseAudio tools detected"
    else
        print_warning "PulseAudio not found"
        echo "To install: sudo apt-get install pulseaudio pavucontrol"
    fi
}

check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader | head -1
        return 0
    else
        print_warning "No NVIDIA GPU detected"
        return 1
    fi
}

install_quick() {
    print_header "Quick Setup - Gemini Live Mode"
    print_info "Installing minimal dependencies (no GPU required)"
    echo ""

    # Create venv
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip --quiet

    # Install minimal dependencies
    echo "Installing core dependencies..."
    pip install numpy sounddevice pyyaml python-dotenv loguru gradio --quiet

    echo "Installing AI libraries..."
    pip install google-generativeai requests beautifulsoup4 duckduckgo-search --quiet

    # Create directories
    mkdir -p data/transcripts data/recordings logs

    # Create .env
    if [ ! -f .env ]; then
        cp .env.example .env
    fi

    print_success "Quick setup complete!"
    echo ""
    print_info "Next steps:"
    echo "  1. Get Gemini API key: https://makersuite.google.com/app/apikey"
    echo "  2. Edit .env: nano .env"
    echo "  3. Add: GEMINI_API_KEY=your_key_here"
    echo "  4. Run: python main.py"
    echo "  5. Select 'Gemini Live (Ultra Fast)' mode in UI"
}

install_full() {
    print_header "Full Setup - Classic Mode"
    print_info "Installing all dependencies (GPU recommended)"
    echo ""

    # Check CUDA
    has_cuda=false
    if check_cuda; then
        has_cuda=true
    fi

    # Create venv
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip --quiet

    # Install PyTorch
    echo "Installing PyTorch..."
    if [ "$has_cuda" = true ]; then
        print_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    else
        print_warning "Installing PyTorch CPU-only (slower performance)"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    fi

    # Install all dependencies
    echo "Installing all dependencies (this may take 5-10 minutes)..."
    pip install -r requirements.txt --quiet

    # Create directories
    mkdir -p data/transcripts data/recordings logs

    # Create .env
    if [ ! -f .env ]; then
        cp .env.example .env
    fi

    # Test GPU
    echo ""
    echo "Testing GPU availability..."
    python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" || true

    print_success "Full setup complete!"
    echo ""
    print_info "Next steps:"
    echo "  1. Get API keys:"
    echo "     - DeepSeek: https://platform.deepseek.com/"
    echo "     - Gemini: https://makersuite.google.com/app/apikey"
    echo "  2. Edit .env: nano .env"
    echo "  3. Add both API keys"
    echo "  4. Run: python main.py"
    echo "  5. Select mode in UI (Classic or Gemini Live)"
}

# Main script
clear
print_header "Meeting Agent - Linux Setup"

echo "This script will help you set up Meeting Agent on Linux."
echo ""
echo "Choose installation mode:"
echo ""
echo -e "${GREEN}1. Quick Setup${NC} - Gemini Live Mode (Recommended)"
echo "   ✓ Fast installation (~2 minutes)"
echo "   ✓ No GPU required"
echo "   ✓ Ultra-fast processing (200-500ms latency)"
echo "   ✓ Minimal dependencies"
echo "   ✗ Requires Gemini API key"
echo ""
echo -e "${BLUE}2. Full Setup${NC} - Classic Mode"
echo "   ✓ Full local control"
echo "   ✓ Offline STT with Faster-Whisper"
echo "   ✓ Customizable models"
echo "   ✓ Both modes available"
echo "   ⚠ Longer installation (~10 minutes)"
echo "   ⚠ GPU recommended for best performance"
echo "   ✗ Requires DeepSeek API key"
echo ""
echo -e "${YELLOW}3. Cancel${NC}"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        check_python
        check_audio
        echo ""
        install_quick
        ;;
    2)
        echo ""
        check_python
        check_audio
        echo ""
        install_full
        ;;
    3)
        echo "Setup cancelled."
        exit 0
        ;;
    *)
        print_error "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
print_header "Setup Complete!"
echo ""
echo -e "${GREEN}To start using Meeting Agent:${NC}"
echo "  1. Activate virtual environment: ${CYAN}source venv/bin/activate${NC}"
echo "  2. Run application: ${CYAN}python main.py${NC}"
echo "  3. Open browser: ${CYAN}http://localhost:7860${NC}"
echo ""
echo "For audio setup, see: docs/LINUX_SETUP.md (coming soon)"
echo "For help: README.md"
echo ""
