"""
Meeting Agent - Main Application
Real-time meeting transcription, translation, and research assistant.
"""

import os
import sys
import yaml
import threading
import queue
import platform
from pathlib import Path
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio.capture import AudioCapture
from transcription.whisper_engine import WhisperTranscriber
from translation.nllb_translator import NLLBTranslator
from ai.deepseek_client import DeepSeekAnalyzer
from ai.gemini_analyzer import GeminiAnalyzer
from ai.gemini_live_client import GeminiLiveClient
from research.web_search import WebResearcher
from ui.gradio_app import MeetingAgentUI


class MeetingAgent:
    """Main Meeting Agent application."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize Meeting Agent.

        Args:
            config_path: Path to configuration file
        """
        # Load environment variables
        load_dotenv()

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config

        # Setup logging
        log_level = config['logging']['level']
        log_file = config['logging']['file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger.remove()
        logger.add(sys.stderr, level=log_level)
        logger.add(log_file, level=log_level, rotation="10 MB")

        logger.info("="*50)
        logger.info("Meeting Agent Starting...")
        logger.info("="*50)

        # Platform detection and warnings
        self._check_platform_compatibility()

        # Create data directories
        os.makedirs(config['storage']['transcripts_dir'], exist_ok=True)
        os.makedirs(config['storage']['recordings_dir'], exist_ok=True)

        # Initialize components
        self.audio_capture = None
        self.transcriber = None
        self.translator = None
        self.analyzer = None
        self.gemini_live = None  # Gemini Live client
        self.researcher = None
        self.ui = None

        # State
        self.is_running = False
        self.transcript_buffer = []
        self.current_language = None
        self.current_mode = "classic"  # classic or live

        # User settings (from UI)
        self.user_settings = {
            'mode': 'classic',  # classic or live
            'analyzer': 'gemini',  # deepseek or gemini (for classic mode)
            'target_lang': 'Turkish',  # Default
            'enable_research': True,
            'deepseek_api_key': None,
            'gemini_api_key': None,
            'whisper_model': 'medium',
            'analysis_interval': 30
        }

        # Processing queue
        self.processing_queue = queue.Queue()

        logger.info("Meeting Agent initialized")

    def _check_platform_compatibility(self):
        """Check platform and warn about compatibility issues."""
        system = platform.system()
        machine = platform.machine()

        logger.info(f"Platform: {system} ({machine})")

        if system == "Darwin":  # macOS
            logger.warning("="*50)
            logger.warning("MACOS DETECTED")
            logger.warning("="*50)
            logger.warning("macOS does NOT support CUDA!")
            logger.warning("")
            logger.warning("Recommendations:")
            logger.warning("  ✓ Use 'Gemini Live' mode (ultra-fast, GPU not needed)")
            logger.warning("  ✗ Avoid 'Classic' mode (will run on CPU, 10x slower)")
            logger.warning("")
            logger.warning("If you must use Classic mode, expect slow performance:")
            logger.warning("  - Whisper will run on CPU (very slow)")
            logger.warning("  - Use 'tiny' or 'base' model only")
            logger.warning("  - 5-10 second delay per transcription")
            logger.warning("="*50)

            # Check for Apple Silicon
            if machine == "arm64":
                logger.info("Apple Silicon detected (M1/M2/M3)")
                logger.info("PyTorch may use MPS acceleration (limited)")

        elif system == "Windows":
            logger.info("Windows detected")
            logger.info("")
            logger.info("Windows Setup Notes:")
            logger.info("  1. Virtual Audio Cable required (VB-Cable or VoiceMeeter)")
            logger.info("  2. For Classic mode: CUDA Toolkit + Visual Studio needed")
            logger.info("  3. Recommended: Use 'Gemini Live' mode (easier setup)")
            logger.info("")
            logger.info("For detailed setup guide, see: docs/WINDOWS_SETUP.md")
            logger.info("Quick install: Run setup_windows.ps1 script")

            # Check for CUDA on Windows
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"✓ CUDA available: {gpu_name}")
                else:
                    logger.warning("⚠ CUDA not available")
                    logger.warning("  Classic mode will run on CPU (slow)")
                    logger.warning("  Recommended: Use 'Gemini Live' mode instead")
            except ImportError:
                logger.info("PyTorch not installed (Classic mode unavailable)")

        elif system == "Linux":
            logger.info("Linux detected - optimal platform ✓")

            # Check for CUDA on Linux
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"✓ CUDA available: {gpu_name} ({vram:.1f}GB)")
                    logger.info("  Both Classic and Gemini Live modes available")
                else:
                    logger.warning("⚠ CUDA not available (GPU not detected)")
                    logger.warning("  Classic mode will run on CPU")
                    logger.warning("  Recommended: Use 'Gemini Live' mode")
            except ImportError:
                logger.info("PyTorch not installed (Classic mode unavailable)")

        else:
            logger.warning(f"Unknown platform: {system}")
            logger.warning("Compatibility not tested, proceed with caution")

    def initialize_components(self):
        """Initialize all components based on mode (lazy loading)."""

        mode = self.user_settings.get('mode', 'classic')
        logger.info(f"Initializing components for {mode.upper()} mode...")

        # Audio capture (always needed)
        audio_config = self.config['audio']
        self.audio_capture = AudioCapture(
            sample_rate=audio_config['sample_rate'],
            channels=audio_config['channels'],
            chunk_duration=audio_config['chunk_duration'],
            device=audio_config['device'],
            callback=self._on_audio_chunk
        )

        if mode == "classic":
            # Classic Mode: Whisper + NLLB + DeepSeek
            self._initialize_classic_mode()
        else:
            # Live Mode: Gemini Live
            self._initialize_live_mode()

        # Web researcher (both modes)
        if self.config['research']['enabled']:
            research_config = self.config['research']
            self.researcher = WebResearcher(
                max_results=research_config['max_results'],
                timeout=research_config['timeout']
            )

        logger.info(f"All components initialized ({mode} mode)")

    def _initialize_classic_mode(self):
        """Initialize Classic mode components (Whisper + NLLB + DeepSeek)."""

        logger.info("Setting up Classic mode pipeline...")

        # Whisper transcriber
        whisper_config = self.config['whisper']
        whisper_model = self.user_settings.get('whisper_model', whisper_config['model_size'])

        self.transcriber = WhisperTranscriber(
            model_size=whisper_model,
            device=whisper_config['device'],
            compute_type=whisper_config['compute_type'],
            language=whisper_config['language'],
            beam_size=whisper_config['beam_size'],
            vad_filter=whisper_config['vad_filter']
        )

        # NLLB translator
        translation_config = self.config['translation']
        self.translator = NLLBTranslator(
            model_name=translation_config['model'],
            device=translation_config['device'],
            target_lang=translation_config['target_lang']
        )

        # AI Analyzer (DeepSeek or Gemini based on user choice)
        analyzer_type = self.user_settings.get('analyzer', 'gemini')

        if analyzer_type == 'deepseek':
            # DeepSeek analyzer
            deepseek_config = self.config['deepseek']
            api_key = self.user_settings.get('deepseek_api_key') or os.getenv('DEEPSEEK_API_KEY')

            if api_key:
                self.analyzer = DeepSeekAnalyzer(
                    api_key=api_key,
                    base_url=deepseek_config['base_url'],
                    model=deepseek_config['model'],
                    temperature=deepseek_config['temperature'],
                    max_tokens=deepseek_config['max_tokens']
                )
                logger.info("DeepSeek analyzer initialized")
            else:
                logger.warning("DEEPSEEK_API_KEY not found, AI analysis disabled")

        else:  # gemini
            # Gemini analyzer (for Classic mode analysis)
            gemini_config = self.config.get('gemini_analyzer', self.config['gemini'])
            api_key = self.user_settings.get('gemini_api_key') or os.getenv('GEMINI_API_KEY')

            if api_key:
                self.analyzer = GeminiAnalyzer(
                    api_key=api_key,
                    model=gemini_config.get('analyzer_model', 'gemini-1.5-flash'),
                    temperature=gemini_config.get('temperature', 0.7),
                    max_tokens=gemini_config.get('max_output_tokens', 2048)
                )
                logger.info("Gemini analyzer initialized (for Classic mode)")
            else:
                logger.warning("GEMINI_API_KEY not found, AI analysis disabled")

    def _initialize_live_mode(self):
        """Initialize Live mode components (Gemini Live only)."""

        logger.info("Setting up Gemini Live mode...")

        # Gemini Live client
        gemini_config = self.config['gemini']
        api_key = self.user_settings.get('gemini_api_key') or os.getenv('GEMINI_API_KEY')

        if api_key:
            target_lang_setting = self.user_settings.get('target_lang', 'Turkish')
            target_lang_name = target_lang_setting.replace(" (Same as source)", "")

            self.gemini_live = GeminiLiveClient(
                api_key=api_key,
                model=gemini_config['model'],
                temperature=gemini_config['generation_config']['temperature'],
                target_language=target_lang_name
            )
            logger.info("Gemini Live client initialized")
        else:
            logger.error("GEMINI_API_KEY not found! Live mode requires Gemini API key.")
            raise ValueError("Gemini API key is required for Live mode")

    def _on_audio_chunk(self, audio_data, sample_rate):
        """Callback for audio chunks (mode-aware)."""
        if not self.is_running:
            return

        mode = self.user_settings.get('mode', 'classic')
        logger.debug(f"Processing audio chunk in {mode} mode: {len(audio_data)} samples")

        try:
            if mode == "classic":
                self._process_classic_mode(audio_data, sample_rate)
            else:
                self._process_live_mode(audio_data, sample_rate)

        except Exception as e:
            logger.error(f"Error processing audio chunk ({mode} mode): {e}", exc_info=True)

    def _process_classic_mode(self, audio_data, sample_rate):
        """Process audio in Classic mode (Whisper + NLLB + DeepSeek)."""

        # Transcribe
        text, language, segments = self.transcriber.transcribe(
            audio_data,
            sample_rate
        )

        if not text.strip():
            logger.debug("Empty transcription, skipping")
            return

        self.current_language = language
        logger.info(f"[CLASSIC] Transcribed [{language}]: {text[:100]}...")

        # Add to buffer
        self.transcript_buffer.append({
            'text': text,
            'language': language,
            'timestamp': time.time()
        })

        # Translate based on user settings
        translation = ""
        target_lang_setting = self.user_settings.get('target_lang', 'Turkish')

        # Map UI language names to codes
        lang_map = {
            'Turkish': 'tr',
            'English': 'en',
            'Auto (Same as source)': None
        }
        target_lang_code = lang_map.get(target_lang_setting, 'tr')

        # Translate if target is different from source
        if self.translator and target_lang_code and language != target_lang_code:
            translation = self.translator.translate_from_whisper_lang(
                text,
                language,
                target_lang_code
            )
            direction = f"{language.upper()} → {target_lang_code.upper()}"
            logger.info(f"[CLASSIC] Translated [{direction}]: {translation[:100]}...")
        else:
            # No translation needed (same language or Auto mode)
            translation = text
            logger.debug(f"Skipping translation: source={language}, target={target_lang_code}")

        # Update UI with transcript and translation
        if self.ui:
            from datetime import datetime
            timestamp = datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
            self.ui.append_transcript(text, timestamp)
            self.ui.append_translation(translation, timestamp)
            self.ui.set_detected_language(language)

        # Queue for analysis
        self.processing_queue.put({
            'type': 'transcript',
            'text': text,
            'translation': translation,
            'language': language
        })

    def _process_live_mode(self, audio_data, sample_rate):
        """Process audio in Live mode (Gemini Live)."""

        # Process with Gemini Live (all-in-one)
        result = self.gemini_live.process_audio(audio_data, sample_rate)

        text = result.get('transcript', '')
        language = result.get('language', 'unknown')
        translation = result.get('translation', '')

        if not text.strip():
            logger.debug("Empty Gemini transcription, skipping")
            return

        self.current_language = language
        logger.info(f"[GEMINI LIVE] Transcribed [{language}]: {text[:100]}...")
        logger.info(f"[GEMINI LIVE] Translated: {translation[:100]}...")

        # Add to buffer
        self.transcript_buffer.append({
            'text': text,
            'language': language,
            'timestamp': time.time()
        })

        # Update UI
        if self.ui:
            from datetime import datetime
            timestamp = datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
            self.ui.append_transcript(text, timestamp)
            self.ui.append_translation(translation, timestamp)
            self.ui.set_detected_language(language)

        # Queue for analysis
        self.processing_queue.put({
            'type': 'transcript',
            'text': text,
            'translation': translation,
            'language': language
        })

    def _processing_worker(self):
        """Background worker for processing tasks (mode-aware)."""
        mode = self.user_settings.get('mode', 'classic')
        logger.info(f"Processing worker started ({mode} mode)")

        accumulated_text = []
        last_analysis_time = 0
        analysis_interval = self.user_settings.get('analysis_interval', 30)

        while self.is_running:
            try:
                # Get item from queue with timeout
                item = self.processing_queue.get(timeout=1)

                if item['type'] == 'transcript':
                    accumulated_text.append(item['text'])

                    # Check if it's time to analyze
                    import time
                    current_time = time.time()

                    if current_time - last_analysis_time >= analysis_interval:
                        if accumulated_text:
                            # Combine accumulated text
                            full_text = " ".join(accumulated_text)

                            # Analyze based on mode
                            if mode == "classic" and self.analyzer:
                                logger.info("[CLASSIC] Performing AI analysis with DeepSeek...")

                                # Extract topics
                                topics = self.analyzer.extract_topics(
                                    full_text,
                                    item['language']
                                )

                                # Generate summary
                                summary = self.analyzer.summarize(
                                    full_text,
                                    item['language']
                                )

                                # Extract action items
                                actions = self.analyzer.extract_action_items(
                                    full_text,
                                    item['language']
                                )

                            elif mode == "live" and self.gemini_live:
                                logger.info("[GEMINI LIVE] Performing AI analysis...")

                                # Use Gemini for analysis
                                analysis = self.gemini_live.analyze_text(
                                    full_text,
                                    item['language']
                                )

                                topics = analysis.get('topics', [])
                                summary = analysis.get('summary', '')
                                actions = analysis.get('actions', [])

                            else:
                                logger.warning(f"No analyzer available for {mode} mode")
                                continue

                            # Update UI with analysis results
                            if self.ui:
                                self.ui.set_analysis(topics, summary, actions)
                                logger.debug("Updated UI with analysis results")

                            # Research topics if enabled by user
                            research_enabled = self.user_settings.get('enable_research', True)
                            research_results = []

                            if self.researcher and topics and research_enabled:
                                # Generate research queries (mode-aware)
                                if mode == "classic" and self.analyzer:
                                    queries = self.analyzer.generate_research_queries(
                                        topics[:3],  # Top 3 topics
                                        "en"
                                    )
                                elif mode == "live" and self.gemini_live:
                                    queries = self.gemini_live.generate_research_queries(topics[:3])
                                else:
                                    queries = []

                                if queries:
                                    research_results = self.researcher.research_multiple(
                                        queries,
                                        fetch_content=False
                                    )

                                    # Update UI with research results
                                    if self.ui:
                                        self.ui.set_research(research_results)
                                        logger.debug("Updated UI with research results")

                                    logger.info(f"Research completed for {len(queries)} queries")
                            elif not research_enabled:
                                logger.debug("Research disabled by user")

                            last_analysis_time = current_time

                            # Clear accumulated text
                            accumulated_text = []

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}", exc_info=True)

        logger.info("Processing worker stopped")

    def start(self, settings=None):
        """
        Start the meeting agent.

        Args:
            settings: Optional dict with user settings from UI
                     {'target_lang': str, 'enable_research': bool}
        """
        if self.is_running:
            logger.warning("Already running")
            return

        # Update settings if provided
        if settings:
            self.user_settings.update(settings)
            self.current_mode = settings.get('mode', 'classic')
            logger.info(f"User settings: {self.user_settings}")
            logger.info(f"Mode: {self.current_mode.upper()}")

        logger.info("Starting Meeting Agent...")

        # Initialize components if not already done
        if not self.transcriber:
            self.initialize_components()

        # Clear UI displays for fresh start
        if self.ui:
            self.ui.clear_displays()

        self.is_running = True

        # Start processing worker
        self.worker_thread = threading.Thread(
            target=self._processing_worker,
            daemon=True
        )
        self.worker_thread.start()

        # Start audio capture
        self.audio_capture.start()

        logger.info("Meeting Agent started")

    def stop(self):
        """Stop the meeting agent."""
        if not self.is_running:
            logger.warning("Not running")
            return

        logger.info("Stopping Meeting Agent...")

        self.is_running = False

        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop()

        # Wait for worker to finish
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=5)

        logger.info("Meeting Agent stopped")

    def save_transcript(self, filename: Optional[str] = None):
        """Save transcript to file."""
        if not self.transcript_buffer:
            logger.warning("No transcript to save")
            return

        import time
        from datetime import datetime

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.txt"

        filepath = Path(self.config['storage']['transcripts_dir']) / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Meeting Transcript\n")
            f.write("=" * 50 + "\n\n")

            for item in self.transcript_buffer:
                timestamp = datetime.fromtimestamp(item['timestamp'])
                f.write(f"[{timestamp.strftime('%H:%M:%S')}] ({item['language']})\n")
                f.write(f"{item['text']}\n\n")

        logger.info(f"Transcript saved to {filepath}")
        return str(filepath)

    def run_ui(self):
        """Run with Gradio UI."""
        ui_config = self.config['ui']

        self.ui = MeetingAgentUI(
            on_start=self.start,
            on_stop=self.stop,
            port=ui_config['port'],
            share=ui_config['share']
        )

        logger.info("Launching UI...")
        self.ui.launch()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Meeting Agent - Real-time meeting assistant")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run without UI (headless mode)"
    )

    args = parser.parse_args()

    # Create agent
    agent = MeetingAgent(config_path=args.config)

    if args.no_ui:
        # Headless mode
        logger.info("Running in headless mode")
        agent.initialize_components()
        agent.start()

        try:
            import time
            logger.info("Press Ctrl+C to stop")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            agent.stop()
            agent.save_transcript()
    else:
        # UI mode
        agent.initialize_components()
        agent.run_ui()


if __name__ == "__main__":
    import time  # Import at module level
    main()
