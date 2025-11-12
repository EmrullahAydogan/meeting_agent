"""
Faster-Whisper transcription engine for real-time speech-to-text.
"""

import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, List, Dict, Tuple
from loguru import logger
import time


class WhisperTranscriber:
    """Real-time speech transcription using Faster-Whisper."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to run on (cuda, cpu)
            compute_type: Computation type (float16, int8, float32)
            language: Language code (None for auto-detect)
            beam_size: Beam size for decoding
            vad_filter: Use VAD filter to remove silence
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        logger.info(f"Loading Whisper model: {model_size} on {device}...")
        start_time = time.time()

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        load_time = time.time() - start_time
        logger.info(f"Whisper model loaded in {load_time:.2f}s")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[str, str, List[Dict]]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            Tuple of (transcribed_text, detected_language, segments)
        """
        if len(audio) == 0:
            return "", "unknown", []

        # Normalize audio to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize amplitude
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        start_time = time.time()

        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=False
        )

        # Collect segments
        segment_list = []
        full_text = []

        for segment in segments:
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": segment.avg_logprob
            }
            segment_list.append(segment_dict)
            full_text.append(segment.text.strip())

        transcription_time = time.time() - start_time
        text = " ".join(full_text)

        logger.info(
            f"Transcribed {len(audio)/sample_rate:.1f}s audio in {transcription_time:.2f}s "
            f"[{info.language}] ({len(text)} chars)"
        )

        return text, info.language, segment_list

    def transcribe_with_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> List[Dict]:
        """
        Transcribe audio with detailed timestamps.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            List of segment dictionaries with timestamps
        """
        _, _, segments = self.transcribe(audio, sample_rate)
        return segments

    def detect_language(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[str, float]:
        """
        Detect the language of audio.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            Tuple of (language_code, confidence)
        """
        if len(audio) == 0:
            return "unknown", 0.0

        # Normalize audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # Detect language
        _, info = self.model.transcribe(
            audio[:sample_rate * 30],  # Use first 30 seconds
            language=None,
            beam_size=1,
            best_of=1
        )

        logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

        return info.language, info.language_probability

    def is_speech(self, audio: np.ndarray, threshold: float = 0.02) -> bool:
        """
        Check if audio contains speech (simple energy-based).

        Args:
            audio: Audio data
            threshold: Energy threshold

        Returns:
            True if audio likely contains speech
        """
        if len(audio) == 0:
            return False

        rms = np.sqrt(np.mean(audio ** 2))
        return rms > threshold


def test_transcriber():
    """Test transcriber with a sample audio file or microphone."""
    import sounddevice as sd

    logger.info("Testing Whisper transcriber...")

    # Initialize transcriber
    transcriber = WhisperTranscriber(
        model_size="base",  # Use smaller model for testing
        device="cuda",
        language=None  # Auto-detect
    )

    # Record 5 seconds of audio
    logger.info("Recording 5 seconds of audio... Please speak!")
    sample_rate = 16000
    duration = 5

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    audio = audio.flatten()
    logger.info(f"Recorded {len(audio)} samples")

    # Transcribe
    text, language, segments = transcriber.transcribe(audio, sample_rate)

    logger.info(f"\nTranscription ({language}):")
    logger.info(f"  {text}")
    logger.info(f"\nSegments:")
    for seg in segments:
        logger.info(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")


if __name__ == "__main__":
    from loguru import logger
    import os
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/transcription_test.log")
    test_transcriber()
