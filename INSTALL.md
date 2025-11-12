# Installation Guide

Detailed installation instructions for Meeting Agent.

## Prerequisites

### Hardware Requirements

**Minimum**:
- 8GB RAM
- 4 CPU cores
- 8GB VRAM (NVIDIA GPU with CUDA support)

**Recommended**:
- 16GB+ RAM
- 8+ CPU cores
- 12-16GB VRAM (NVIDIA RTX 3060 or better)

**Without GPU**:
- System will work on CPU but will be significantly slower
- Not recommended for real-time use

### Software Requirements

- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, or macOS 11+

## Installation Steps

### 1. System Dependencies

#### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt-get update

# Install Python and essential tools
sudo apt-get install -y python3 python3-pip python3-venv

# Install audio tools
sudo apt-get install -y pulseaudio pavucontrol portaudio19-dev

# Install build tools (for some Python packages)
sudo apt-get install -y build-essential

# Install CUDA (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads
```

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install audio tools
brew install portaudio

# Install BlackHole for system audio capture
brew install blackhole-2ch
```

#### Windows

1. Install Python 3.10+ from https://www.python.org/downloads/
2. Install Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
4. Install VB-Cable: https://vb-audio.com/Cable/

### 2. Clone Repository

```bash
git clone <repository-url>
cd meeting_agent
```

### 3. Automated Setup (Linux/macOS)

```bash
./setup.sh
```

This will:
- Create virtual environment
- Install PyTorch with CUDA
- Install all dependencies
- Create necessary directories
- Create .env file

### 4. Manual Setup (All Platforms)

#### Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### Install PyTorch

Visit https://pytorch.org/get-started/locally/ and select your configuration.

**Example for CUDA 11.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Example for CPU only**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configuration

#### Create Environment File

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

Get DeepSeek API key from: https://platform.deepseek.com/

#### Configure Settings

Edit `config/settings.yaml` to customize:

```yaml
# Adjust model sizes based on your GPU
whisper:
  model_size: "medium"  # Use "small" or "base" for lower VRAM

translation:
  model: "facebook/nllb-200-distilled-600M"  # Smaller model

# Adjust for CPU-only mode
whisper:
  device: "cpu"
  compute_type: "int8"

translation:
  device: "cpu"
```

## Audio Setup

### Linux (PulseAudio)

```bash
# Start PulseAudio
pulseaudio --start

# Load loopback module (to capture system audio)
pactl load-module module-loopback

# Use pavucontrol to route audio
pavucontrol
```

In pavucontrol:
1. Go to "Recording" tab
2. Find Python/Meeting Agent
3. Set it to "Monitor of <your output device>"

### macOS (BlackHole)

1. Install BlackHole: `brew install blackhole-2ch`
2. Open Audio MIDI Setup
3. Create Multi-Output Device:
   - Include your speakers + BlackHole
4. Set Multi-Output Device as system output
5. In Meeting Agent, select BlackHole as input

### Windows (VB-Cable)

1. Install VB-Cable from https://vb-audio.com/Cable/
2. Set "CABLE Input" as your system's default playback device
3. Set "CABLE Output" as your recording device in Meeting Agent

## Verification

### Test GPU

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Should output: `CUDA available: True`

### Test Audio

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Should list your audio devices.

### Test Whisper

```bash
python src/transcription/whisper_engine.py
```

Should download Whisper model and test transcription.

### Test DeepSeek API

```bash
python src/ai/deepseek_client.py
```

Should connect to DeepSeek API and run tests.

## Troubleshooting

### PyTorch CUDA Issues

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Audio Issues

**No audio devices**:
```bash
# Linux
sudo apt-get install pulseaudio
pulseaudio --start

# Check devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**Permission denied**:
```bash
# Add user to audio group (Linux)
sudo usermod -a -G audio $USER
# Log out and back in
```

### Model Download Issues

If models fail to download:

```bash
# Set Hugging Face cache directory
export HF_HOME=/path/with/space
export TRANSFORMERS_CACHE=/path/with/space

# Or set in Python before imports
import os
os.environ['HF_HOME'] = '/path/with/space'
```

### Memory Issues

Reduce model sizes in `config/settings.yaml`:

```yaml
whisper:
  model_size: "base"  # Smaller model
  compute_type: "int8"  # Lower precision

translation:
  model: "facebook/nllb-200-distilled-600M"  # Smaller model
```

## Next Steps

After successful installation:

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **Test with a short meeting**

3. **Review logs** in `logs/meeting_agent.log`

4. **Adjust settings** in `config/settings.yaml` based on performance

5. **Read the full README.md** for usage instructions

## Getting Help

- Check logs: `tail -f logs/meeting_agent.log`
- Verify configuration: `cat config/settings.yaml`
- Test components individually (see README.md)
- Open an issue on GitHub with logs and system info

## System Information

When reporting issues, include:

```bash
# Python version
python --version

# PyTorch info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# GPU info
nvidia-smi

# OS info
uname -a  # Linux/macOS
systeminfo  # Windows
```
