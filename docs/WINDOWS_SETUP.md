# Windows Setup Guide

Complete guide for setting up Meeting Agent on Windows.

## Prerequisites

### 1. Python 3.10+
Download from [python.org](https://www.python.org/downloads/)
- ✅ Check "Add Python to PATH" during installation

### 2. NVIDIA GPU (for Classic Mode)
- NVIDIA GPU with CUDA support
- At least 4GB VRAM recommended

---

## Option A: Quick Setup (Gemini Live Mode - RECOMMENDED)

**Best for Windows users** - No GPU setup required, ultra-fast, minimal dependencies.

### Step 1: Install Virtual Audio Cable

Choose one:

#### VB-Audio Virtual Cable (Recommended)
1. Download from [vb-audio.com](https://vb-audio.com/Cable/)
2. Extract and run `VBCABLE_Setup_x64.exe` as Administrator
3. Restart your computer

#### VoiceMeeter Banana (Free Alternative)
1. Download from [vb-audio.com](https://vb-audio.com/Voicemeeter/banana.htm)
2. Install and configure virtual audio routing

### Step 2: Install Python Dependencies

```powershell
# Open PowerShell as Administrator
cd meeting_agent

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install minimal dependencies for Gemini Live
pip install numpy sounddevice pyyaml python-dotenv loguru gradio
pip install google-generativeai

# Install PyAudio (Windows wheel)
pip install pipwin
pipwin install pyaudio
```

### Step 3: Configure API Keys

```powershell
# Copy example env file
copy .env.example .env

# Edit .env and add your Gemini API key
notepad .env
```

Add:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 4: Run

```powershell
python main.py
```

Select **"Gemini Live (Ultra Fast)"** mode in the UI.

---

## Option B: Full Setup (Classic Mode)

Only choose this if you need offline STT or want full control.

### Step 1: Install CUDA Toolkit

1. **Check GPU Compatibility**
   ```powershell
   nvidia-smi
   ```
   If this works, you have an NVIDIA GPU.

2. **Install Visual Studio Build Tools**
   - Download [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++"
   - ~6GB download

3. **Install CUDA Toolkit 12.x**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - ~3GB download
   - Choose Express Installation
   - Restart computer after installation

4. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (requires free account)
   - Extract and copy files to CUDA directory:
     ```powershell
     # Copy to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\
     ```

### Step 2: Install PyTorch with CUDA

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Install PyTorch with CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install All Dependencies

```powershell
# Install from requirements.txt (this may take 10-15 minutes)
pip install -r requirements.txt

# Fix PyAudio if needed
pip install pipwin
pipwin install pyaudio
```

### Step 4: Install Virtual Audio Cable

Same as Option A, Step 1.

### Step 5: Configure API Keys

```powershell
copy .env.example .env
notepad .env
```

Add:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 6: Test CUDA

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Should print: `CUDA available: True`

### Step 7: Run

```powershell
python main.py
```

Select **"Classic (Whisper + DeepSeek)"** mode in the UI.

---

## Audio Routing Setup

### For Google Meet / Zoom / Teams

1. **Start your virtual audio cable** (VBCABLE or VoiceMeeter)

2. **In Windows Sound Settings**:
   - Right-click speaker icon → "Sound settings"
   - Output: Select "CABLE Input" or VoiceMeeter
   - Input: Keep your microphone

3. **In Meeting App** (Meet/Zoom/Teams):
   - Audio Output: Set to "CABLE Input"
   - Audio Input: Set to your real microphone

4. **In Meeting Agent**:
   - The app will automatically capture from "CABLE Output"

5. **To hear audio yourself**:
   - Use VoiceMeeter to route CABLE Output → Your speakers
   - Or use Windows "Listen to this device" feature

---

## Common Issues & Solutions

### Issue 1: PyAudio Installation Fails

```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution**:
```powershell
pip install pipwin
pipwin install pyaudio
```

Or download precompiled wheel:
```powershell
# For Python 3.11 (64-bit)
pip install https://download.lfd.uci.edu/pythonlibs/archived/PyAudio-0.2.13-cp311-cp311-win_amd64.whl
```

### Issue 2: CUDA Not Detected

**Solution 1**: Check CUDA installation
```powershell
nvcc --version
```

**Solution 2**: Reinstall PyTorch
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: No Audio Captured

**Solution**:
1. Check virtual audio cable is running
2. Verify audio routing in Sound Settings
3. Test with Audacity or similar tool first
4. Check Meeting Agent is listening to correct device:
   - Edit `config/settings.yaml`
   - Set `audio.device` to correct index (try 0, 1, 2, etc.)

### Issue 4: DLL Load Failed

```
ImportError: DLL load failed while importing _something
```

**Solution**:
Install Visual C++ Redistributable:
- Download [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Install and restart

### Issue 5: Out of Memory (Classic Mode)

**Solution**:
Use smaller Whisper model in UI:
- Change to "small" or "base" instead of "medium"
- Or switch to Gemini Live mode

---

## Performance Comparison

| Mode          | Setup Time | First Run  | Latency | Accuracy |
|---------------|-----------|------------|---------|----------|
| Gemini Live   | 5 mins    | Instant    | 0.3s    | Excellent |
| Classic Mode  | 30-60 mins| 2-3 mins   | 3-5s    | Excellent |

**Recommendation**: Start with Gemini Live mode, switch to Classic only if you need offline processing.

---

## Troubleshooting

### Get Help
1. Check logs in `logs/meeting_agent.log`
2. Enable debug mode in `config/settings.yaml`:
   ```yaml
   logging:
     level: DEBUG
   ```
3. Report issues with log file

### System Requirements

**Minimum (Gemini Live)**:
- CPU: Any modern processor
- RAM: 4GB
- GPU: Not required
- Internet: Required

**Recommended (Classic Mode)**:
- CPU: Intel i5 / AMD Ryzen 5 or better
- RAM: 16GB
- GPU: NVIDIA GPU with 6GB+ VRAM
- Internet: Required for DeepSeek API

---

## Quick Start Script

Save this as `setup_windows.ps1`:

```powershell
# Meeting Agent - Windows Quick Setup
Write-Host "Meeting Agent - Windows Setup" -ForegroundColor Green

# Check Python
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found! Install from python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
python -m venv venv
.\venv\Scripts\activate

# Install minimal dependencies (Gemini Live)
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install numpy sounddevice pyyaml python-dotenv loguru gradio google-generativeai

# Try to install PyAudio
Write-Host "`nInstalling PyAudio..." -ForegroundColor Yellow
pip install pipwin
pipwin install pyaudio

# Copy env file
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "`n.env file created. Edit it to add your API keys:" -ForegroundColor Green
    Write-Host "  GEMINI_API_KEY=your_key_here" -ForegroundColor Cyan
}

Write-Host "`n" -NoNewline
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Install Virtual Audio Cable (VB-Cable)" -ForegroundColor White
Write-Host "  2. Edit .env file and add GEMINI_API_KEY" -ForegroundColor White
Write-Host "  3. Run: python main.py" -ForegroundColor White
Write-Host "  4. Select 'Gemini Live' mode in UI" -ForegroundColor White
Write-Host "`nFor Classic mode with GPU, see docs/WINDOWS_SETUP.md" -ForegroundColor Gray
```

Run with:
```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

---

## Still Having Issues?

**Use Gemini Live mode** - it bypasses 90% of Windows setup complexity:
- No CUDA needed
- No GPU needed
- No PyTorch needed
- Just Python + Virtual Audio Cable + Gemini API key

The performance is actually **better** than Classic mode (200-500ms vs 3-5s latency).
