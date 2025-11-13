"""
Gemini Live client for real-time audio processing.
Handles transcription, translation, and analysis in a single API call.
"""

import os
import numpy as np
import google.generativeai as genai
from typing import Optional, Dict, List, Tuple
from loguru import logger
import time
import base64


class GeminiLiveClient:
    """Gemini Live API client for ultra-fast audio processing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        target_language: str = "Turkish"
    ):
        """
        Initialize Gemini Live client.

        Args:
            api_key: Gemini API key (from env if None)
            model: Model name
            temperature: Sampling temperature
            target_language: Target language for translation
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or parameters")

        self.model_name = model
        self.temperature = temperature
        self.target_language = target_language

        # Configure API
        genai.configure(api_key=self.api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

        logger.info(f"GeminiLiveClient initialized: {model}")

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        target_lang: str = None
    ) -> Dict:
        """
        Process audio: transcribe, detect language, and translate.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio
            target_lang: Target language for translation (override default)

        Returns:
            Dict with 'transcript', 'language', 'translation'
        """
        if len(audio) == 0:
            return {
                'transcript': '',
                'language': 'unknown',
                'translation': '',
                'segments': []
            }

        target = target_lang or self.target_language

        try:
            start_time = time.time()

            # Convert audio to bytes
            # Gemini expects base64 encoded audio
            audio_bytes = self._audio_to_bytes(audio, sample_rate)

            # Create prompt for transcription and translation
            prompt = f"""
You are a real-time meeting assistant. Process this audio and provide:

1. TRANSCRIPTION: Transcribe the audio exactly as spoken
2. LANGUAGE: Detect the language (respond with 2-letter code: en, tr, de, etc.)
3. TRANSLATION: Translate to {target} (if source language is different)

Respond in this exact JSON format:
{{
  "transcript": "exact transcription here",
  "language": "en",
  "translation": "translation to {target} here"
}}

IMPORTANT:
- If source language is the same as target, copy transcript to translation
- Be accurate and concise
- Use proper punctuation
"""

            # Upload audio
            audio_file = genai.upload_file(
                path=self._save_temp_audio(audio_bytes),
                mime_type="audio/wav"
            )

            # Generate response
            response = self.model.generate_content(
                [prompt, audio_file],
                request_options={"timeout": 30}
            )

            # Parse response
            result = self._parse_response(response.text)

            processing_time = time.time() - start_time
            logger.info(
                f"Gemini processed {len(audio)/sample_rate:.1f}s audio in {processing_time:.2f}s "
                f"[{result['language']}]"
            )

            return result

        except Exception as e:
            logger.error(f"Gemini Live error: {e}")
            return {
                'transcript': '',
                'language': 'unknown',
                'translation': '',
                'segments': []
            }

    def analyze_text(
        self,
        text: str,
        language: str = "auto"
    ) -> Dict:
        """
        Analyze text for topics, summary, and action items.

        Args:
            text: Text to analyze
            language: Language of text

        Returns:
            Dict with 'topics', 'summary', 'actions'
        """
        if not text.strip():
            return {
                'topics': [],
                'summary': '',
                'actions': []
            }

        try:
            start_time = time.time()

            prompt = f"""
Analyze this meeting transcript and extract:

1. TOPICS: 3-5 main topics discussed (one per line)
2. SUMMARY: Concise 2-3 sentence summary
3. ACTIONS: Action items and tasks (numbered list)

Transcript language: {language}
Transcript:
{text}

Respond in this exact JSON format:
{{
  "topics": ["topic1", "topic2", "topic3"],
  "summary": "summary here",
  "actions": ["action 1", "action 2"]
}}
"""

            response = self.model.generate_content(
                prompt,
                request_options={"timeout": 20}
            )

            result = self._parse_response(response.text)

            analysis_time = time.time() - start_time
            logger.info(f"Gemini analyzed text in {analysis_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return {
                'topics': [],
                'summary': '',
                'actions': []
            }

    def generate_research_queries(self, topics: List[str]) -> List[str]:
        """
        Generate research queries from topics.

        Args:
            topics: List of topics

        Returns:
            List of search queries
        """
        if not topics:
            return []

        topics_text = "\n".join(f"- {topic}" for topic in topics)

        prompt = f"""
Generate 2-3 concise search queries for researching these topics:

{topics_text}

Return ONLY the search queries, one per line, no numbering or extra text.
"""

        try:
            response = self.model.generate_content(prompt)
            queries = [
                line.strip()
                for line in response.text.strip().split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
            return queries[:3]

        except Exception as e:
            logger.error(f"Query generation error: {e}")
            return []

    def _audio_to_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy audio to WAV bytes."""
        import io
        import wave

        # Normalize to int16
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

        return buffer.getvalue()

    def _save_temp_audio(self, audio_bytes: bytes) -> str:
        """Save audio bytes to temporary file."""
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_bytes)
        temp_file.close()
        return temp_file.name

    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini JSON response."""
        import json
        import re

        try:
            # Try to extract JSON from response
            # Sometimes Gemini wraps JSON in ```json blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text

            result = json.loads(json_str)
            return result

        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Response text: {response_text}")
            # Return empty result
            return {
                'transcript': '',
                'language': 'unknown',
                'translation': '',
                'topics': [],
                'summary': '',
                'actions': []
            }


def test_gemini_live():
    """Test Gemini Live client."""
    logger.info("Testing Gemini Live client...")

    # Initialize client
    client = GeminiLiveClient(
        model="gemini-2.0-flash-exp",
        target_language="Turkish"
    )

    # Test with dummy audio (silence)
    sample_rate = 16000
    duration = 3
    audio = np.zeros(sample_rate * duration, dtype=np.float32)

    logger.info("Testing audio processing...")
    result = client.process_audio(audio, sample_rate)
    logger.info(f"Result: {result}")

    # Test analysis
    test_text = """
    Good morning everyone. Today we need to discuss the Q4 product roadmap.
    First, we should finalize the mobile app features. John, can you prepare
    the technical specifications by Friday? Also, we need to review the budget
    for the marketing campaign.
    """

    logger.info("\nTesting text analysis...")
    analysis = client.analyze_text(test_text, "en")
    logger.info(f"Topics: {analysis.get('topics')}")
    logger.info(f"Summary: {analysis.get('summary')}")
    logger.info(f"Actions: {analysis.get('actions')}")


if __name__ == "__main__":
    from loguru import logger
    import os
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/gemini_test.log")
    test_gemini_live()
