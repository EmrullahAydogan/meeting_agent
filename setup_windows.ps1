# Meeting Agent - Windows Quick Setup Script
# This script installs Meeting Agent with Gemini Live mode (recommended for Windows)

param(
    [switch]$FullInstall = $false,
    [switch]$Help = $false
)

function Write-ColorOutput($ForegroundColor, $Message) {
    Write-Host $Message -ForegroundColor $ForegroundColor
}

function Write-Header($Message) {
    Write-Host ""
    Write-ColorOutput "Cyan" "═══════════════════════════════════════════════════════"
    Write-ColorOutput "Cyan" " $Message"
    Write-ColorOutput "Cyan" "═══════════════════════════════════════════════════════"
}

function Test-Command($Command) {
    try {
        if (Get-Command $Command -ErrorAction Stop) {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

# Show help
if ($Help) {
    Write-ColorOutput "Green" "Meeting Agent - Windows Setup Script"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\setup_windows.ps1              # Quick install (Gemini Live mode)"
    Write-Host "  .\setup_windows.ps1 -FullInstall # Full install with GPU support"
    Write-Host "  .\setup_windows.ps1 -Help        # Show this help"
    Write-Host ""
    Write-Host "Recommended: Use quick install (Gemini Live) for best Windows experience"
    exit 0
}

# Banner
Write-Header "Meeting Agent - Windows Setup"
Write-ColorOutput "Yellow" "Real-time Meeting Transcription, Translation & Analysis"
Write-Host ""

# Check Python
Write-ColorOutput "Yellow" "[1/6] Checking Python installation..."
if (-not (Test-Command "python")) {
    Write-ColorOutput "Red" "✗ Python not found!"
    Write-ColorOutput "White" "  Please install Python 3.10+ from https://www.python.org/downloads/"
    Write-ColorOutput "White" "  Make sure to check 'Add Python to PATH' during installation"
    exit 1
}

$pythonVersion = python --version 2>&1
Write-ColorOutput "Green" "✓ $pythonVersion found"

# Check pip
if (-not (Test-Command "pip")) {
    Write-ColorOutput "Red" "✗ pip not found!"
    Write-ColorOutput "White" "  Please reinstall Python with pip included"
    exit 1
}
Write-ColorOutput "Green" "✓ pip found"

# Check virtual audio cable
Write-Host ""
Write-ColorOutput "Yellow" "[2/6] Checking Virtual Audio Cable..."
Write-ColorOutput "White" "  Virtual Audio Cable is required to capture system audio"

$audioDrivers = Get-WmiObject Win32_SoundDevice | Select-Object -ExpandProperty Name
$hasVirtualCable = $audioDrivers -match "CABLE|VoiceMeeter|Virtual"

if ($hasVirtualCable) {
    Write-ColorOutput "Green" "✓ Virtual audio device found"
} else {
    Write-ColorOutput "Red" "✗ No virtual audio cable detected"
    Write-ColorOutput "White" ""
    Write-ColorOutput "White" "  Please install one of the following:"
    Write-ColorOutput "Cyan" "    1. VB-Audio Virtual Cable (Recommended)"
    Write-ColorOutput "White" "       https://vb-audio.com/Cable/"
    Write-ColorOutput "Cyan" "    2. VoiceMeeter Banana (Free)"
    Write-ColorOutput "White" "       https://vb-audio.com/Voicemeeter/banana.htm"
    Write-Host ""

    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y") {
        Write-ColorOutput "Yellow" "Setup cancelled. Install virtual audio cable first."
        exit 0
    }
}

# Create virtual environment
Write-Host ""
Write-ColorOutput "Yellow" "[3/6] Creating virtual environment..."

if (Test-Path "venv") {
    Write-ColorOutput "Cyan" "  Virtual environment already exists, skipping..."
} else {
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Red" "✗ Failed to create virtual environment"
        exit 1
    }
    Write-ColorOutput "Green" "✓ Virtual environment created"
}

# Activate virtual environment
Write-ColorOutput "Cyan" "  Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-ColorOutput "Cyan" "  Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host ""
Write-ColorOutput "Yellow" "[4/6] Installing dependencies..."

if ($FullInstall) {
    Write-ColorOutput "Cyan" "  Full installation mode (GPU support)"
    Write-ColorOutput "White" "  This will take 10-30 minutes depending on your connection..."
    Write-Host ""

    # Check CUDA
    Write-ColorOutput "Cyan" "  Checking NVIDIA GPU..."
    if (Test-Command "nvidia-smi") {
        Write-ColorOutput "Green" "✓ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    } else {
        Write-ColorOutput "Yellow" "⚠ No NVIDIA GPU detected or drivers not installed"
        Write-ColorOutput "White" "  Classic mode will run on CPU (very slow)"
        $response = Read-Host "Continue with full install? (y/n)"
        if ($response -ne "y") {
            Write-ColorOutput "Yellow" "Switching to quick install (Gemini Live)"
            $FullInstall = $false
        }
    }

    if ($FullInstall) {
        # Install PyTorch with CUDA
        Write-ColorOutput "Cyan" "  Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

        # Install all requirements
        Write-ColorOutput "Cyan" "  Installing all requirements..."
        pip install -r requirements.txt --quiet
    }
}

if (-not $FullInstall) {
    Write-ColorOutput "Cyan" "  Quick installation mode (Gemini Live - recommended)"
    Write-ColorOutput "White" "  Installing minimal dependencies..."
    Write-Host ""

    # Install minimal dependencies
    $packages = @(
        "numpy>=1.24.0",
        "sounddevice>=0.4.6",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.2",
        "gradio>=4.0.0",
        "google-generativeai>=0.3.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "duckduckgo-search>=4.0.0"
    )

    foreach ($package in $packages) {
        $pkgName = $package -replace ">=.*", ""
        Write-Host "  Installing $pkgName..." -NoNewline
        pip install $package --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Green" " ✓"
        } else {
            Write-ColorOutput "Red" " ✗"
        }
    }
}

# Install PyAudio
Write-Host ""
Write-ColorOutput "Yellow" "[5/6] Installing PyAudio..."
Write-ColorOutput "Cyan" "  Attempting pipwin installation..."

pip install pipwin --quiet
pipwin install pyaudio

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "Yellow" "⚠ pipwin failed, trying alternative method..."
    pip install pyaudio

    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Red" "✗ PyAudio installation failed"
        Write-ColorOutput "White" "  You may need to install Visual C++ Build Tools"
        Write-ColorOutput "White" "  See docs/WINDOWS_SETUP.md for manual installation"
        Write-Host ""

        $response = Read-Host "Continue without PyAudio? (y/n)"
        if ($response -ne "y") {
            exit 1
        }
    } else {
        Write-ColorOutput "Green" "✓ PyAudio installed"
    }
} else {
    Write-ColorOutput "Green" "✓ PyAudio installed via pipwin"
}

# Setup .env file
Write-Host ""
Write-ColorOutput "Yellow" "[6/6] Setting up configuration..."

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-ColorOutput "Green" "✓ .env file created"
    Write-ColorOutput "Yellow" "⚠ You need to add your API keys to .env file"
} else {
    Write-ColorOutput "Cyan" "  .env file already exists, skipping..."
}

# Test installation
Write-Host ""
Write-ColorOutput "Yellow" "Testing installation..."

$testScript = @"
import sys
try:
    import numpy
    import sounddevice
    import yaml
    import gradio
    print('✓ Core packages OK')

    if '$FullInstall' == 'True':
        import torch
        print(f'✓ PyTorch OK (CUDA: {torch.cuda.is_available()})')
    else:
        import google.generativeai
        print('✓ Gemini API OK')

    sys.exit(0)
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
"@

$testScript | python -
$testSuccess = $LASTEXITCODE -eq 0

Write-Host ""
Write-ColorOutput "Green" "═══════════════════════════════════════════════════════"
Write-ColorOutput "Green" " Installation Complete!"
Write-ColorOutput "Green" "═══════════════════════════════════════════════════════"
Write-Host ""

if ($testSuccess) {
    Write-ColorOutput "Green" "✓ All tests passed"
} else {
    Write-ColorOutput "Yellow" "⚠ Some tests failed, but installation may still work"
}

Write-Host ""
Write-ColorOutput "Cyan" "Next Steps:"
Write-Host ""

if (-not $hasVirtualCable) {
    Write-ColorOutput "Yellow" "1. Install Virtual Audio Cable:"
    Write-ColorOutput "White" "   Download from: https://vb-audio.com/Cable/"
    Write-Host ""
}

Write-ColorOutput "Yellow" "2. Configure API Keys:"
Write-ColorOutput "White" "   Edit .env file and add:"
if ($FullInstall) {
    Write-ColorOutput "Cyan" "     DEEPSEEK_API_KEY=your_key_here"
}
Write-ColorOutput "Cyan" "     GEMINI_API_KEY=your_key_here"
Write-Host ""

Write-ColorOutput "Yellow" "3. Run Meeting Agent:"
Write-ColorOutput "White" "   .\venv\Scripts\Activate.ps1"
Write-ColorOutput "White" "   python main.py"
Write-Host ""

if ($FullInstall) {
    Write-ColorOutput "Cyan" "   Select 'Classic (Whisper + DeepSeek)' mode in UI"
} else {
    Write-ColorOutput "Cyan" "   Select 'Gemini Live (Ultra Fast)' mode in UI"
}

Write-Host ""
Write-ColorOutput "White" "For detailed setup guide, see: docs/WINDOWS_SETUP.md"
Write-ColorOutput "White" "For troubleshooting, check: logs/meeting_agent.log"
Write-Host ""

# Open .env file for editing
$response = Read-Host "Open .env file now to add API keys? (y/n)"
if ($response -eq "y") {
    notepad .env
}

Write-Host ""
Write-ColorOutput "Green" "Happy meeting transcribing! 🎙️"
