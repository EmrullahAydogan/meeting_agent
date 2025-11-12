"""
Audio capture module for recording system audio from video conferences.
Supports multiple platforms (Linux, Windows, macOS).
"""

import numpy as np
import sounddevice as sd
import queue
import threading
from typing import Optional, Callable
from loguru import logger


class AudioCapture:
    """Captures system audio in real-time."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: int = 30,
        device: Optional[int] = None,
        callback: Optional[Callable] = None
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_duration: Duration of audio chunks in seconds
            device: Audio device index (None for default)
            callback: Callback function to process audio chunks
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.device = device
        self.callback = callback

        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self._thread = None

        logger.info(f"AudioCapture initialized: {sample_rate}Hz, {channels}ch")

    def list_devices(self):
        """List all available audio devices."""
        devices = sd.query_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            logger.info(f"  [{i}] {device['name']} - "
                       f"In:{device['max_input_channels']} Out:{device['max_output_channels']}")
        return devices

    def _audio_callback(self, indata, frames, time, status):
        """Internal callback for audio stream."""
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Convert to mono if needed
        if self.channels == 1 and indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.copy()

        # Add to queue
        self.audio_queue.put(audio_data)

    def _process_audio(self):
        """Process audio chunks from queue."""
        buffer = []
        chunk_samples = int(self.sample_rate * self.chunk_duration)

        while self.is_recording:
            try:
                # Get audio from queue
                audio_chunk = self.audio_queue.get(timeout=1)
                buffer.append(audio_chunk)

                # Check if we have enough samples
                total_samples = sum(len(chunk) for chunk in buffer)

                if total_samples >= chunk_samples:
                    # Concatenate buffer
                    audio_data = np.concatenate(buffer)

                    # Take chunk and keep remainder
                    chunk = audio_data[:chunk_samples]
                    remainder = audio_data[chunk_samples:]

                    # Reset buffer with remainder
                    buffer = [remainder] if len(remainder) > 0 else []

                    # Process chunk
                    if self.callback:
                        try:
                            self.callback(chunk, self.sample_rate)
                        except Exception as e:
                            logger.error(f"Error in audio callback: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")

    def start(self):
        """Start audio capture."""
        if self.is_recording:
            logger.warning("Already recording")
            return

        logger.info("Starting audio capture...")
        self.is_recording = True

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            callback=self._audio_callback
        )
        self.stream.start()

        # Start processing thread
        self._thread = threading.Thread(target=self._process_audio, daemon=True)
        self._thread.start()

        logger.info("Audio capture started")

    def stop(self):
        """Stop audio capture."""
        if not self.is_recording:
            logger.warning("Not recording")
            return

        logger.info("Stopping audio capture...")
        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        logger.info("Audio capture stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def get_default_device():
    """Get default audio input device."""
    return sd.default.device[0]


def test_audio_capture():
    """Test audio capture functionality."""
    def audio_callback(audio_data, sample_rate):
        rms = np.sqrt(np.mean(audio_data**2))
        logger.info(f"Received audio chunk: {len(audio_data)} samples, RMS: {rms:.4f}")

    logger.info("Testing audio capture for 10 seconds...")

    capture = AudioCapture(
        sample_rate=16000,
        channels=1,
        chunk_duration=5,
        callback=audio_callback
    )

    capture.list_devices()

    with capture:
        import time
        time.sleep(10)

    logger.info("Test complete")


if __name__ == "__main__":
    from loguru import logger
    logger.add("logs/audio_test.log")
    test_audio_capture()
