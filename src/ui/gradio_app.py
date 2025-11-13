"""
Gradio UI for Meeting Agent.
Displays real-time transcription, translation, and research results.
All settings managed through UI - no .env file needed!
"""

import gradio as gr
from typing import Optional, Callable, Dict, Any
from loguru import logger
import time
import threading


class MeetingAgentUI:
    """Gradio UI for Meeting Agent with centralized settings management."""

    def __init__(
        self,
        on_start: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
        port: int = 7860,
        share: bool = False
    ):
        """
        Initialize Meeting Agent UI.

        Args:
            on_start: Callback when recording starts
            on_stop: Callback when recording stops
            port: Port to run on
            share: Whether to create public link
        """
        self.on_start = on_start
        self.on_stop = on_stop
        self.port = port
        self.share = share

        self.is_recording = False
        self.app = None

        # Shared state for real-time updates
        self.shared_state = {
            'transcript': '',
            'translation': '',
            'detected_lang': 'Not detected yet',
            'topics': '',
            'summary': '',
            'actions': '',
            'research': '<p>Research results will appear here...</p>',
            'last_update': time.time()
        }

        # Lock for thread-safe access
        self.state_lock = threading.Lock()

        logger.info("MeetingAgentUI initialized with centralized settings")

    def update_state(self, key: str, value: Any):
        """
        Thread-safe update of shared state.

        Args:
            key: State key to update
            value: New value
        """
        with self.state_lock:
            self.shared_state[key] = value
            self.shared_state['last_update'] = time.time()

    def get_state(self) -> Dict[str, Any]:
        """
        Thread-safe get of shared state.

        Returns:
            Copy of current state
        """
        with self.state_lock:
            return self.shared_state.copy()

    def create_interface(self):
        """Create Gradio interface with Settings tab."""

        with gr.Blocks(title="Meeting Agent - AI Meeting Assistant", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🎙️ Meeting Agent")
            gr.Markdown("**Real-time AI Meeting Assistant** - Transcribe, translate, and analyze your meetings")

            with gr.Row():
                # Left Column: Control Panel
                with gr.Column(scale=1):
                    gr.Markdown("### 🎬 Control")

                    start_btn = gr.Button(
                        "▶️ Start Recording",
                        variant="primary",
                        size="lg"
                    )
                    stop_btn = gr.Button(
                        "⏹️ Stop Recording",
                        variant="stop",
                        size="lg"
                    )

                    status_text = gr.Textbox(
                        label="Status",
                        value="⚙️ Ready - Configure settings and press Start",
                        interactive=False,
                        lines=2
                    )

                    gr.Markdown("---")
                    gr.Markdown("### ℹ️ Quick Info")

                    detected_lang = gr.Textbox(
                        label="Detected Language",
                        value="Not detected yet",
                        interactive=False
                    )

                    gr.Markdown(
                        """
                        **Tip:** Go to ⚙️ Settings tab to configure API keys and preferences before starting!
                        """
                    )

                # Right Column: Tabbed Content
                with gr.Column(scale=3):
                    with gr.Tabs() as main_tabs:
                        # Tab 1: Live View
                        with gr.TabItem("📺 Live View"):
                            gr.Markdown("### Real-time Results")

                            with gr.Tabs():
                                with gr.TabItem("📝 Original"):
                                    transcript_box = gr.Textbox(
                                        label="Transcription (Original Language)",
                                        lines=12,
                                        max_lines=20,
                                        interactive=False,
                                        placeholder="Start recording to see live transcription...",
                                        autoscroll=True
                                    )

                                with gr.TabItem("🌍 Translation"):
                                    translation_box = gr.Textbox(
                                        label="Translation (Target Language)",
                                        lines=12,
                                        max_lines=20,
                                        interactive=False,
                                        placeholder="Translated text will appear here...",
                                        autoscroll=True
                                    )

                                with gr.TabItem("🤖 Analysis"):
                                    with gr.Row():
                                        with gr.Column():
                                            gr.Markdown("#### 🎯 Topics")
                                            topics_box = gr.Textbox(
                                                label="Key Topics Discussed",
                                                lines=6,
                                                interactive=False,
                                                placeholder="AI-extracted topics will appear here..."
                                            )

                                        with gr.Column():
                                            gr.Markdown("#### 📊 Summary")
                                            summary_box = gr.Textbox(
                                                label="Meeting Summary",
                                                lines=6,
                                                interactive=False,
                                                placeholder="AI-generated summary will appear here..."
                                            )

                                    gr.Markdown("#### ✅ Action Items")
                                    actions_box = gr.Textbox(
                                        label="Tasks & To-Dos",
                                        lines=6,
                                        interactive=False,
                                        placeholder="Action items will be listed here..."
                                    )

                                with gr.TabItem("🔍 Research"):
                                    research_box = gr.HTML(
                                        label="Web Research Results",
                                        value="<p style='color: #6b7280;'>Research results will appear here when enabled...</p>"
                                    )

                        # Tab 2: Settings (ALL CONFIGURATION HERE!)
                        with gr.TabItem("⚙️ Settings"):
                            gr.Markdown("### Configuration Center")
                            gr.Markdown("*Configure all settings here - No .env file needed!*")

                            # Processing Mode Section
                            gr.Markdown("---")
                            gr.Markdown("## 🔧 Processing Mode")

                            mode_selector = gr.Radio(
                                choices=[
                                    "Classic (Whisper + AI)",
                                    "Gemini Live (Ultra Fast)"
                                ],
                                value="Classic (Whisper + AI)",
                                label="Select Processing Mode",
                                info="Classic: Offline STT + AI Analysis | Live: All-in-one cloud processing (200-500ms latency)"
                            )

                            analyzer_selector = gr.Radio(
                                choices=["Gemini", "DeepSeek"],
                                value="Gemini",
                                label="AI Analyzer (for Classic Mode only)",
                                info="Gemini: Powerful, single API key | DeepSeek: Ultra-cheap (~$0.27/1M tokens)"
                            )

                            # API Keys Section
                            gr.Markdown("---")
                            gr.Markdown("## 🔑 API Keys")
                            gr.Markdown("*Enter your API keys below. Get them from:*")
                            gr.Markdown("- **Gemini**: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)")
                            gr.Markdown("- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com/)")

                            gemini_key = gr.Textbox(
                                label="🤖 Gemini API Key",
                                type="password",
                                placeholder="AIza... (Required for Gemini Live or Gemini Analyzer)",
                                value="",
                                info="Used for: Gemini Live mode AND/OR Classic mode with Gemini analyzer"
                            )

                            deepseek_key = gr.Textbox(
                                label="💰 DeepSeek API Key (Optional)",
                                type="password",
                                placeholder="sk-... (Only needed if using DeepSeek analyzer in Classic mode)",
                                value="",
                                info="Only required if you select DeepSeek as analyzer in Classic mode"
                            )

                            # Model Settings Section
                            gr.Markdown("---")
                            gr.Markdown("## 🎛️ Model Settings")

                            whisper_model = gr.Dropdown(
                                choices=["tiny", "base", "small", "medium", "large-v3"],
                                value="medium",
                                label="Whisper Model (Classic Mode Only)",
                                info="tiny=fastest/lowest quality, large-v3=slowest/best quality (requires 8GB+ VRAM)"
                            )

                            analysis_interval = gr.Slider(
                                minimum=15,
                                maximum=120,
                                value=30,
                                step=15,
                                label="AI Analysis Interval (seconds)",
                                info="How often to run analysis (topics, summary, actions)"
                            )

                            # Language & Translation Section
                            gr.Markdown("---")
                            gr.Markdown("## 🌍 Language & Translation")

                            target_lang = gr.Dropdown(
                                choices=[
                                    "Turkish",
                                    "English",
                                    "Auto (Same as source)"
                                ],
                                value="Turkish",
                                label="Translation Target Language",
                                info="Select target language for translation (Auto = no translation)"
                            )

                            # Features Section
                            gr.Markdown("---")
                            gr.Markdown("## 🚀 Features")

                            enable_research = gr.Checkbox(
                                label="Enable Web Research",
                                value=True,
                                info="Automatically research discussed topics using DuckDuckGo"
                            )

                            # Save Settings Button
                            gr.Markdown("---")
                            save_settings_btn = gr.Button(
                                "💾 Save Settings",
                                variant="secondary",
                                size="sm"
                            )
                            settings_status = gr.Textbox(
                                label="Settings Status",
                                value="Settings loaded. Click 'Save Settings' to apply changes.",
                                interactive=False,
                                lines=1
                            )

            # Event Handlers
            def start_recording(
                mode_val, analyzer_val, target_lang_val, enable_research_val,
                gemini_key_val, deepseek_key_val, whisper_model_val, analysis_interval_val
            ):
                """Handle start recording button click."""
                if self.on_start:
                    # Parse mode
                    mode = "live" if "Gemini Live" in mode_val else "classic"

                    # Validate API keys
                    if mode == "live" and not gemini_key_val.strip():
                        return "❌ Error: Gemini API key required for Gemini Live mode! Go to Settings tab."

                    if mode == "classic":
                        if analyzer_val == "Gemini" and not gemini_key_val.strip():
                            return "❌ Error: Gemini API key required for Gemini analyzer! Go to Settings tab."
                        if analyzer_val == "DeepSeek" and not deepseek_key_val.strip():
                            return "❌ Error: DeepSeek API key required for DeepSeek analyzer! Go to Settings tab."

                    # Pass all settings to callback
                    settings = {
                        'mode': mode,
                        'analyzer': analyzer_val.lower(),  # 'deepseek' or 'gemini'
                        'target_lang': target_lang_val,
                        'enable_research': enable_research_val,
                        'deepseek_api_key': deepseek_key_val.strip() if deepseek_key_val else None,
                        'gemini_api_key': gemini_key_val.strip() if gemini_key_val else None,
                        'whisper_model': whisper_model_val,
                        'analysis_interval': int(analysis_interval_val)
                    }

                    try:
                        self.on_start(settings)
                        self.is_recording = True

                        mode_emoji = "⚡" if mode == "live" else "🔴"
                        mode_text = "Gemini Live" if mode == "live" else f"Classic ({analyzer_val})"
                        return f"{mode_emoji} Recording in progress - {mode_text} mode active!"
                    except Exception as e:
                        return f"❌ Error starting: {str(e)}"

                return "❌ Error: No start callback configured"

            def stop_recording():
                """Handle stop recording button click."""
                if self.on_stop:
                    self.on_stop()
                self.is_recording = False
                return "⏹️ Stopped - Ready to start again"

            def save_settings_action(
                mode_val, analyzer_val, target_lang_val, enable_research_val,
                gemini_key_val, deepseek_key_val, whisper_model_val, analysis_interval_val
            ):
                """Handle save settings button click."""
                # Validate settings
                mode = "live" if "Gemini Live" in mode_val else "classic"
                warnings = []

                if not gemini_key_val.strip() and not deepseek_key_val.strip():
                    warnings.append("⚠️ No API keys entered - you'll need at least one to use the app")

                if mode == "live" and not gemini_key_val.strip():
                    warnings.append("⚠️ Gemini Live mode requires Gemini API key")

                if mode == "classic" and analyzer_val == "Gemini" and not gemini_key_val.strip():
                    warnings.append("⚠️ Gemini analyzer requires Gemini API key")

                if mode == "classic" and analyzer_val == "DeepSeek" and not deepseek_key_val.strip():
                    warnings.append("⚠️ DeepSeek analyzer requires DeepSeek API key")

                if warnings:
                    return "\n".join(warnings)
                else:
                    return "✅ Settings saved! Ready to start recording."

            def update_displays():
                """Update all display components with latest state."""
                state = self.get_state()
                return (
                    state['transcript'],
                    state['translation'],
                    state['detected_lang'],
                    state['topics'],
                    state['summary'],
                    state['actions'],
                    state['research']
                )

            # Wire up event handlers
            start_btn.click(
                fn=start_recording,
                inputs=[
                    mode_selector, analyzer_selector, target_lang, enable_research,
                    gemini_key, deepseek_key, whisper_model, analysis_interval
                ],
                outputs=status_text
            )

            stop_btn.click(
                fn=stop_recording,
                outputs=status_text
            )

            save_settings_btn.click(
                fn=save_settings_action,
                inputs=[
                    mode_selector, analyzer_selector, target_lang, enable_research,
                    gemini_key, deepseek_key, whisper_model, analysis_interval
                ],
                outputs=settings_status
            )

            # Real-time updates - poll every 2 seconds
            update_timer = gr.Timer(value=2, active=True)
            update_timer.tick(
                fn=update_displays,
                outputs=[
                    transcript_box,
                    translation_box,
                    detected_lang,
                    topics_box,
                    summary_box,
                    actions_box,
                    research_box
                ]
            )

            # Store references
            self.components = {
                'transcript': transcript_box,
                'translation': translation_box,
                'topics': topics_box,
                'summary': summary_box,
                'actions': actions_box,
                'research': research_box,
                'status': status_text,
                'detected_lang': detected_lang,
                'target_lang': target_lang,
                'enable_research': enable_research
            }

        self.app = app
        return app

    def append_transcript(self, text: str, timestamp: str = None):
        """
        Append text to transcript (accumulative).

        Args:
            text: Text to append
            timestamp: Optional timestamp string
        """
        with self.state_lock:
            if timestamp:
                entry = f"[{timestamp}] {text}\n\n"
            else:
                entry = f"{text}\n\n"
            self.shared_state['transcript'] += entry

    def append_translation(self, text: str, timestamp: str = None):
        """
        Append translation (accumulative).

        Args:
            text: Translation text to append
            timestamp: Optional timestamp string
        """
        with self.state_lock:
            if timestamp:
                entry = f"[{timestamp}] {text}\n\n"
            else:
                entry = f"{text}\n\n"
            self.shared_state['translation'] += entry

    def set_detected_language(self, language: str):
        """
        Update detected language display.

        Args:
            language: Language code (e.g., 'en', 'tr')
        """
        lang_names = {
            'en': 'English',
            'tr': 'Turkish (Türkçe)',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic'
        }
        display_name = lang_names.get(language, language.upper())
        self.update_state('detected_lang', f"🗣️ {display_name}")

    def set_analysis(self, topics: list, summary: str, actions: list):
        """
        Update analysis results.

        Args:
            topics: List of topics
            summary: Summary text
            actions: List of action items
        """
        # Format topics
        if topics:
            topics_text = "\n".join(f"• {topic}" for topic in topics)
        else:
            topics_text = "No topics extracted yet..."

        # Format actions
        if actions:
            actions_text = "\n".join(f"{i}. {action}" for i, action in enumerate(actions, 1))
        else:
            actions_text = "No action items yet..."

        self.update_state('topics', topics_text)
        self.update_state('summary', summary if summary else "No summary yet...")
        self.update_state('actions', actions_text)

    def set_research(self, research_data: list):
        """
        Update research results.

        Args:
            research_data: List of research result dictionaries
        """
        if not research_data:
            html = "<p style='color: #6b7280;'>No research results yet...</p>"
        else:
            html = "<div style='font-family: sans-serif;'>"
            for result in research_data:
                query = result.get('query', 'Unknown')
                results = result.get('results', [])

                html += f"<h3 style='color: #2563eb;'>🔍 {query}</h3>"

                if results:
                    for i, item in enumerate(results[:3], 1):  # Top 3 results
                        title = item.get('title', 'No title')
                        url = item.get('url', '#')
                        snippet = item.get('snippet', '')

                        html += f"""
                        <div style='margin: 10px 0; padding: 10px; background: #f3f4f6; border-radius: 5px;'>
                            <strong>{i}. <a href='{url}' target='_blank' style='color: #1d4ed8;'>{title}</a></strong>
                            <p style='margin: 5px 0; color: #6b7280; font-size: 0.9em;'>{snippet}</p>
                        </div>
                        """
                else:
                    html += "<p style='color: #9ca3af;'>No results found</p>"

                html += "<hr style='margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;'>"

            html += "</div>"

        self.update_state('research', html)

    def clear_displays(self):
        """Clear all display content."""
        with self.state_lock:
            self.shared_state = {
                'transcript': '',
                'translation': '',
                'detected_lang': 'Not detected yet',
                'topics': '',
                'summary': '',
                'actions': '',
                'research': '<p>Research results will appear here...</p>',
                'last_update': time.time()
            }

    def launch(self):
        """Launch Gradio app."""
        if not self.app:
            self.create_interface()

        logger.info(f"Launching Gradio app on port {self.port}...")

        self.app.launch(
            server_port=self.port,
            share=self.share,
            inbrowser=True
        )

    def queue_update(self, update_type: str, data: str):
        """Queue an update to the UI."""
        # This will be used by the main app to push updates
        pass


def create_simple_demo():
    """Create a simple demo UI."""

    def start_callback(settings):
        logger.info(f"Start button clicked with settings: {settings}")

    def stop_callback():
        logger.info("Stop button clicked")

    ui = MeetingAgentUI(
        on_start=start_callback,
        on_stop=stop_callback,
        port=7860,
        share=False
    )

    ui.create_interface()
    return ui


def test_ui():
    """Test UI with dummy data."""
    logger.info("Testing Gradio UI...")

    ui = create_simple_demo()
    ui.launch()


if __name__ == "__main__":
    from loguru import logger
    import os
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/ui_test.log")
    test_ui()
