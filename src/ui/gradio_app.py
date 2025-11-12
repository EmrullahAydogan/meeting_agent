"""
Gradio UI for Meeting Agent.
Displays real-time transcription, translation, and research results.
"""

import gradio as gr
from typing import Optional, Callable
from loguru import logger
import time


class MeetingAgentUI:
    """Gradio UI for Meeting Agent."""

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

        logger.info("MeetingAgentUI initialized")

    def create_interface(self):
        """Create Gradio interface."""

        with gr.Blocks(title="Meeting Agent", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🎙️ Meeting Agent")
            gr.Markdown("Real-time transcription, translation, and research for video conferences")

            with gr.Row():
                with gr.Column(scale=1):
                    # Control panel
                    gr.Markdown("### Control")
                    start_btn = gr.Button("▶️ Start Recording", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop Recording", variant="stop", size="lg")
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False
                    )

                    # Settings
                    gr.Markdown("### Settings")

                    # Language info
                    detected_lang = gr.Textbox(
                        label="Detected Language",
                        value="Not detected yet",
                        interactive=False
                    )

                    target_lang = gr.Dropdown(
                        choices=["Turkish", "English", "Auto (Same as source)"],
                        value="Turkish",
                        label="Translation Target",
                        info="Language to translate to"
                    )

                    enable_research = gr.Checkbox(
                        label="Enable Research",
                        value=True
                    )

                with gr.Column(scale=2):
                    # Main display
                    gr.Markdown("### Live Transcription")

                    with gr.Tabs():
                        with gr.TabItem("Original"):
                            transcript_box = gr.Textbox(
                                label="Transcription",
                                lines=10,
                                max_lines=15,
                                interactive=False,
                                placeholder="Transcription will appear here..."
                            )

                        with gr.TabItem("Translation"):
                            translation_box = gr.Textbox(
                                label="Translation",
                                lines=10,
                                max_lines=15,
                                interactive=False,
                                placeholder="Translation will appear here..."
                            )

                        with gr.TabItem("Analysis"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### Topics")
                                    topics_box = gr.Textbox(
                                        label="Key Topics",
                                        lines=5,
                                        interactive=False,
                                        placeholder="Topics will appear here..."
                                    )

                                with gr.Column():
                                    gr.Markdown("#### Summary")
                                    summary_box = gr.Textbox(
                                        label="Summary",
                                        lines=5,
                                        interactive=False,
                                        placeholder="Summary will appear here..."
                                    )

                            gr.Markdown("#### Action Items")
                            actions_box = gr.Textbox(
                                label="Action Items",
                                lines=5,
                                interactive=False,
                                placeholder="Action items will appear here..."
                            )

                        with gr.TabItem("Research"):
                            research_box = gr.HTML(
                                label="Research Results",
                                value="<p>Research results will appear here...</p>"
                            )

            # Event handlers
            def start_recording(target_lang_val, enable_research_val):
                if self.on_start:
                    # Pass settings to callback
                    settings = {
                        'target_lang': target_lang_val,
                        'enable_research': enable_research_val
                    }
                    self.on_start(settings)
                self.is_recording = True
                return "🔴 Recording..."

            def stop_recording():
                if self.on_stop:
                    self.on_stop()
                self.is_recording = False
                return "⏹️ Stopped"

            start_btn.click(
                fn=start_recording,
                inputs=[target_lang, enable_research],
                outputs=status_text
            )

            stop_btn.click(
                fn=stop_recording,
                outputs=status_text
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

    def update_transcript(self, text: str):
        """Update transcript display."""
        if self.app and 'transcript' in self.components:
            return text

    def update_translation(self, text: str):
        """Update translation display."""
        if self.app and 'translation' in self.components:
            return text

    def update_detected_language(self, language: str):
        """Update detected language display."""
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
        if self.app and 'detected_lang' in self.components:
            return f"🗣️ {display_name}"
        return display_name

    def update_analysis(self, topics: str, summary: str, actions: str):
        """Update analysis display."""
        return topics, summary, actions

    def update_research(self, html: str):
        """Update research display."""
        if self.app and 'research' in self.components:
            return html

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

    def start_callback():
        logger.info("Start button clicked")

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
