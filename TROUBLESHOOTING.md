# 🔧 Troubleshooting Guide

Common issues and solutions for Meeting Agent installation and usage.

---

## Table of Contents

- [Installation Issues](#installation-issues)
  - [PyAudio Compilation Error](#pyaudio-compilation-error)
  - [PyTorch/Transformers Compatibility](#pytorchtransformers-compatibility)
  - [CUDA Not Detected](#cuda-not-detected)
- [Runtime Issues](#runtime-issues)
  - [Audio Not Capturing](#audio-not-capturing)
  - [Model Loading Errors](#model-loading-errors)
  - [Out of Memory](#out-of-memory)
- [Platform-Specific Issues](#platform-specific-issues)
  - [Linux](#linux)
  - [Windows](#windows)
  - [macOS](#macos)

---

## Installation Issues

### PyAudio Compilation Error

**Error:**
```
fatal error: portaudio.h: No such file or directory
error: Failed building wheel for pyaudio
```

**Cause:** PortAudio development headers are not installed on your system.

**Solution (Linux):**
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev build-essential
```

**Solution (macOS):**
```bash
brew install portaudio
```

**Solution (Windows):**
Use pre-built wheel:
```bash
pip install pipwin
pipwin install pyaudio
```

---

### PyTorch/Transformers Compatibility

**Error:**
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with
`weights_only=True`, we now require users to upgrade torch to at least v2.6
in order to use the function.
See https://nvd.nist.gov/vuln/detail/CVE-2025-32434
```

**Cause:**
- PyTorch ≤2.5.1 has a critical security vulnerability (CVE-2025-32434)
- Transformers 4.57+ enforces PyTorch 2.6+ for safety
- PyTorch 2.6 is not yet stable (only nightly builds available)
- NLLB-200 models use old pickle format (`.bin` files)

**Solutions (Choose One):**

#### Option 1: Use Gemini Live Mode (Recommended)
Avoid the issue entirely by using Gemini Live mode:
```bash
# Quick Setup - No PyTorch needed
./setup.sh
# Select: 1. Quick Setup - Gemini Live Mode
```

#### Option 2: Downgrade Transformers (Temporary Fix)
```bash
pip install transformers==4.47.0
```

⚠️ **Security Note:** This uses an older version without the security check.

#### Option 3: Manual Patch (Advanced Users)
If you need Classic mode with current versions:

1. Locate the file:
   ```bash
   venv/lib/python3.X/site-packages/transformers/utils/import_utils.py
   ```

2. Find the `check_torch_load_is_safe()` function (around line 1645)

3. Replace with:
   ```python
   def check_torch_load_is_safe() -> None:
       # PATCHED: Temporarily bypassing PyTorch 2.6 requirement
       # WARNING: Only use with trusted models from HuggingFace
       # TODO: Remove when PyTorch 2.6 stable is released
       pass
   ```

4. Save and restart the application

⚠️ **Security Warning:**
- Only use models from trusted sources (HuggingFace official)
- Never load models from unknown sources
- Upgrade to PyTorch 2.6 when stable

**References:**
- [GitHub Issue #38464](https://github.com/huggingface/transformers/issues/38464)
- [CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)

---

### CUDA Not Detected

**Error:**
```
CUDA not available
```

**Possible Causes:**
1. NVIDIA drivers not installed
2. CUDA toolkit not installed
3. Wrong PyTorch version (CPU-only)

**Diagnosis:**
```bash
# Check if GPU is detected
nvidia-smi

# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions:**

**Linux:**
```bash
# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall

# Install CUDA toolkit (if needed)
sudo apt install nvidia-cuda-toolkit

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Windows:**
1. Install NVIDIA drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
2. Install [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-downloads)
3. Reinstall PyTorch with CUDA support

---

## Runtime Issues

### Audio Not Capturing

**Symptoms:**
- No transcription appears
- Status shows "Recording" but nothing happens
- Empty transcript boxes

**Solutions:**

#### Linux:
```bash
# Check if PulseAudio is running
pulseaudio --check
pulseaudio --start

# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test audio capture
python src/audio/capture.py
```

#### Configure Virtual Audio (for video conferences):
```bash
# Load loopback module
pactl load-module module-loopback

# Use pavucontrol to route audio
pavucontrol
```

#### Windows:
1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
2. Set virtual cable as recording device
3. Route meeting audio through virtual cable

---

### Model Loading Errors

**Error:**
```
OSError: Unable to load model weights
```

**Solutions:**

1. **Clear HuggingFace cache:**
   ```bash
   rm -rf ~/.cache/huggingface/hub/
   ```

2. **Re-download models:**
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/nllb-200-distilled-600M')"
   ```

3. **Check disk space:**
   ```bash
   df -h ~/.cache/huggingface/
   ```
   Models require ~4GB of space.

---

### Out of Memory

**Error:**
```
CUDA out of memory
RuntimeError: CUDA error: out of memory
```

**Solutions:**

1. **Use smaller Whisper model:**
   - In UI Settings: Change Whisper model to `small` or `base`
   - Edit `config/settings.yaml`:
     ```yaml
     whisper:
       model_size: "small"  # Instead of "medium"
     ```

2. **Use CPU instead of GPU:**
   ```yaml
   whisper:
     device: "cpu"
   translation:
     device: "cpu"
   ```

3. **Switch to Gemini Live mode:**
   - No local models needed
   - No GPU required
   - Ultra-fast processing

4. **Close other GPU applications:**
   ```bash
   # Check GPU usage
   nvidia-smi

   # Kill GPU processes if needed
   ```

---

## Platform-Specific Issues

### Linux

#### PulseAudio Not Running
```bash
# Start PulseAudio
pulseaudio --start

# Check status
systemctl --user status pulseaudio
```

#### Permission Errors
```bash
# Add user to audio group
sudo usermod -aG audio $USER

# Re-login for changes to take effect
```

#### Dependencies Missing
```bash
# Install all required system packages
sudo apt-get update
sudo apt-get install -y \
    pulseaudio \
    pavucontrol \
    portaudio19-dev \
    python3-dev \
    build-essential \
    ffmpeg
```

---

### Windows

#### Virtual Audio Cable Issues

**Problem:** VB-Cable not working

**Solutions:**
1. Run VB-Cable installer as Administrator
2. Restart computer after installation
3. Check in Sound Settings:
   - Recording devices: VB-Cable Output should appear
   - Playback devices: VB-Cable Input should appear

#### CUDA Installation on Windows

**Required:**
1. Visual Studio Build Tools (C++ workload)
2. CUDA Toolkit 12.1+
3. cuDNN (optional, for better performance)

**Installation Order:**
1. Visual Studio Build Tools first
2. CUDA Toolkit second
3. Restart computer
4. Install PyTorch with CUDA

---

### macOS

#### No CUDA Support
**Limitation:** macOS does not support NVIDIA CUDA.

**Solution:** Use Gemini Live mode only
- Fast cloud-based processing
- No GPU required
- Works perfectly on M1/M2/M3 chips

#### BlackHole Audio Setup
```bash
# Install BlackHole
brew install blackhole-2ch

# Create Multi-Output Device
# Open Audio MIDI Setup
# Create new Multi-Output Device
# Add: Built-in Output + BlackHole 2ch
```

---

## Getting Help

If you still have issues:

1. **Check logs:**
   ```bash
   cat logs/meeting_agent.log
   ```

2. **Enable debug logging:**
   Edit `config/settings.yaml`:
   ```yaml
   logging:
     level: "DEBUG"
   ```

3. **Test individual components:**
   ```bash
   # Test audio
   python src/audio/capture.py

   # Test Whisper
   python src/transcription/whisper_engine.py

   # Test translation
   python src/translation/nllb_translator.py

   # Test UI
   python src/ui/gradio_app.py
   ```

4. **Create GitHub issue:**
   - Include error messages
   - Include logs (last 50 lines)
   - Include system info (OS, Python version, GPU)
   - Include steps to reproduce

---

## Common Error Messages

### "API key not found"
➜ Go to ⚙️ Settings tab in UI and enter your API key

### "No module named 'torch'"
➜ Run setup script or manually install: `pip install torch`

### "CUDA out of memory"
➜ Use smaller model or switch to Gemini Live mode

### "Audio device not found"
➜ Check virtual audio cable installation and configuration

### "Model download failed"
➜ Check internet connection and retry

### "Import error: No module named 'transformers'"
➜ Run: `pip install transformers`

---

## Prevention Tips

1. **Always use virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   ```

2. **Keep packages updated:**
   ```bash
   pip install --upgrade pip
   pip install --upgrade -r requirements.txt
   ```

3. **Check system requirements before installing:**
   - Python 3.10+
   - 16GB RAM (Classic mode)
   - 8GB+ VRAM (Classic mode with GPU)
   - Stable internet connection

4. **Use recommended mode for your platform:**
   - Linux with GPU → Classic mode
   - Linux without GPU → Gemini Live
   - Windows → Gemini Live (easier setup)
   - macOS → Gemini Live (only option)

---

**Last Updated:** November 13, 2025

For more help, visit: https://github.com/EmrullahAydogan/meeting_agent/issues
