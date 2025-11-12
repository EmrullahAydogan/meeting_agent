# 🎙️ Meeting Agent

**Real-time AI Meeting Assistant** for video conferences (Google Meet, Zoom, Microsoft Teams)

Automatically transcribes, translates, analyzes, and researches topics discussed in your meetings.

## ✨ Features

- **🎤 Real-time Transcription**: Speech-to-text using Faster-Whisper (GPU-accelerated)
- **🌍 Live Translation**: Multilingual support with NLLB-200 (TR ↔ EN and 200+ languages)
- **🤖 AI Analysis**: Topic extraction, summarization, and action items via DeepSeek
- **🔍 Smart Research**: Automatic web research on discussed topics
- **👥 Speaker Diarization**: Identify who is speaking (optional)
- **💻 Modern UI**: Clean Gradio interface with real-time updates
- **💰 Cost-Effective**: Mostly open-source, ~$2-5/month for DeepSeek API

## 🏗️ Architecture

```
Video Conference (Meet/Zoom/Teams)
    ↓
System Audio Capture (PulseAudio/Virtual Cable)
    ↓
Faster-Whisper (GPU) → Real-time Transcription
    ↓
NLLB-200 (GPU) → Translation
    ↓
DeepSeek AI → Analysis (Topics, Summary, Actions)
    ↓
Web Research → DuckDuckGo + Content Extraction
    ↓
Gradio UI → Display Results
```

## 📋 Requirements

### Hardware
- **GPU**: CUDA-capable GPU recommended (8GB+ VRAM)
  - Minimum: 8GB VRAM (medium Whisper model)
  - Recommended: 12-16GB VRAM (large Whisper model)
  - Without GPU: CPU mode available (slower)

### Software
- **OS**: Linux (primary), Windows, macOS
- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU acceleration)

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd meeting_agent
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your DeepSeek API key
nano .env
```

Get your DeepSeek API key from: https://platform.deepseek.com/

```env
DEEPSEEK_API_KEY=your_api_key_here
```

### 4. Configure Audio (Linux)

For capturing system audio on Linux:

```bash
# Install PulseAudio tools
sudo apt-get install pulseaudio pavucontrol

# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Configure audio loopback (optional)
pactl load-module module-loopback
```

For Windows: Install [VB-Cable](https://vb-audio.com/Cable/) or similar virtual audio device.

For macOS: Install [BlackHole](https://existential.audio/blackhole/) or [Loopback](https://rogueamoeba.com/loopback/).

### 5. Run Application

```bash
# With UI (recommended)
python main.py

# Headless mode
python main.py --no-ui

# Custom config
python main.py --config my_config.yaml
```

The UI will open in your browser at `http://localhost:7860`

## 📖 Usage

### Basic Workflow

1. **Start Application**: Run `python main.py`
2. **Configure Audio**: Set your system audio as input
3. **Start Meeting**: Join your Google Meet/Zoom/Teams meeting
4. **Click Start**: Begin recording in the UI
5. **View Results**: See real-time transcription, translation, and analysis
6. **Stop Recording**: Save transcript when done

### UI Tabs

- **Original**: Raw transcription in detected language
- **Translation**: Translated text (configurable target language)
- **Analysis**: Topics, summary, and action items
- **Research**: Web research results on discussed topics

## ⚙️ Configuration

Edit `config/settings.yaml` to customize:

### Whisper Settings
```yaml
whisper:
  model_size: "medium"  # tiny, base, small, medium, large-v3
  device: "cuda"        # cuda or cpu
  compute_type: "float16"
  language: null        # null for auto-detect, or "tr", "en"
```

### Translation Settings
```yaml
translation:
  model: "facebook/nllb-200-distilled-600M"
  target_lang: "tr"  # Default translation target
```

### DeepSeek Settings
```yaml
deepseek:
  model: "deepseek-chat"  # or deepseek-reasoner
  temperature: 0.7
  max_tokens: 2000
```

### Research Settings
```yaml
research:
  enabled: true
  max_results: 3
  search_engine: "duckduckgo"
```

## 🧪 Testing Individual Modules

Test each component separately:

```bash
# Test audio capture
python src/audio/capture.py

# Test Whisper transcription
python src/transcription/whisper_engine.py

# Test NLLB translation
python src/translation/nllb_translator.py

# Test DeepSeek analyzer
python src/ai/deepseek_client.py

# Test web research
python src/research/web_search.py

# Test UI
python src/ui/gradio_app.py
```

## 📁 Project Structure

```
meeting_agent/
├── config/
│   └── settings.yaml          # Configuration file
├── src/
│   ├── audio/
│   │   └── capture.py         # Audio capture module
│   ├── transcription/
│   │   └── whisper_engine.py  # Whisper STT
│   ├── translation/
│   │   └── nllb_translator.py # NLLB translation
│   ├── ai/
│   │   └── deepseek_client.py # DeepSeek AI client
│   ├── research/
│   │   └── web_search.py      # Web research
│   └── ui/
│       └── gradio_app.py      # Gradio UI
├── data/                      # Saved transcripts
├── logs/                      # Application logs
├── main.py                    # Main application
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 💡 Tips & Best Practices

### Audio Quality
- Use good quality microphone
- Minimize background noise
- Adjust system audio levels

### Performance
- Use GPU for best performance
- Close other GPU-intensive applications
- Use `medium` Whisper model for balance of speed/accuracy

### Cost Optimization
- DeepSeek API is very cheap (~$0.27/1M tokens)
- Adjust `analysis_interval` in code to reduce API calls
- Disable research if not needed

### Privacy
- All processing happens locally except DeepSeek API calls
- No data is stored on external servers
- Transcripts saved locally only

## 🐛 Troubleshooting

### Audio Not Capturing

**Linux**:
```bash
# Check PulseAudio
pulseaudio --check
pulseaudio --start

# Use pavucontrol to route audio
pavucontrol
```

**Windows**: Ensure virtual audio cable is installed and selected as recording device.

### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

- Use smaller Whisper model (`small` or `base`)
- Reduce `chunk_duration` in settings
- Use `int8` compute type instead of `float16`

### DeepSeek API Errors

- Verify API key in `.env`
- Check API quota/credits
- Review logs in `logs/meeting_agent.log`

## 🔧 Advanced Configuration

### Custom Whisper Model

```python
# In config/settings.yaml
whisper:
  model_size: "large-v3"  # Best accuracy
  compute_type: "int8"    # Lower memory usage
```

### Multiple Languages

```python
# Auto-detect and translate to Turkish
translation:
  source_lang: "auto"
  target_lang: "tur_Latn"
```

### Custom Research Engine

Implement your own in `src/research/web_search.py`:
- SearXNG (self-hosted)
- Brave Search API
- Google Custom Search

## 📊 Performance Benchmarks

### Whisper Model Comparison

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 1GB | 32x | ⭐⭐ |
| base | 1GB | 16x | ⭐⭐⭐ |
| small | 2GB | 6x | ⭐⭐⭐⭐ |
| medium | 5GB | 2x | ⭐⭐⭐⭐⭐ |
| large-v3 | 10GB | 1x | ⭐⭐⭐⭐⭐⭐ |

### Cost Estimate

**Monthly Usage** (20 hours of meetings):
- DeepSeek API: ~$2-5
- Compute: ~$0 (local GPU)
- Total: **~$2-5/month**

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Speaker diarization integration
- More translation models
- Additional LLM backends
- Mobile app interface
- Recording playback

## 📝 License

MIT License - feel free to use and modify

## 🙏 Acknowledgments

Built with:
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb)
- [DeepSeek](https://www.deepseek.com/)
- [Gradio](https://gradio.app/)
- [PyAnnote](https://github.com/pyannote/pyannote-audio)

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check logs in `logs/meeting_agent.log`
- Review configuration in `config/settings.yaml`

---

Made with ❤️ for better meetings
