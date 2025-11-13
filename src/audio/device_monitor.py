"""
Audio device monitor for real-time audio level detection.
Used by UI to show live audio levels for device selection.
"""

import numpy as np
import sounddevice as sd
import threading
import time
from typing import Optional, Callable
from loguru import logger


class AudioDeviceMonitor:
    """Monitors audio levels from a specific device in real-time."""

    def __init__(self, device: Optional[int] = None, sample_rate: int = 16000):
        """
        Initialize audio device monitor.

        Args:
            device: Audio device index (None for default)
            sample_rate: Sample rate for monitoring
        """
        self.device = device
        self.sample_rate = sample_rate
        self.is_monitoring = False
        self.stream = None
        self.current_level = 0.0  # 0-100 scale
        self._lock = threading.Lock()

        logger.debug(f"AudioDeviceMonitor initialized for device {device}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback to process audio and calculate level."""
        if status:
            logger.warning(f"Monitor stream status: {status}")

        try:
            # Calculate RMS (Root Mean Square) level
            rms = np.sqrt(np.mean(indata**2))

            # Convert to percentage (0-100)
            # Assuming maximum RMS of 1.0 for normalization
            level = min(100, rms * 100 * 10)  # Amplify by 10 for visibility

            with self._lock:
                self.current_level = level

        except Exception as e:
            logger.error(f"Error in monitor callback: {e}")

    def start(self):
        """Start monitoring audio levels."""
        if self.is_monitoring:
            logger.warning("Already monitoring")
            return

        try:
            # Detect device sample rate
            if self.device is not None:
                device_info = sd.query_devices(self.device, 'input')
            else:
                device_info = sd.query_devices(kind='input')

            actual_sample_rate = int(device_info['default_samplerate'])

            logger.info(f"Starting audio monitor on device {self.device} at {actual_sample_rate}Hz")

            self.is_monitoring = True

            # Start monitoring stream
            self.stream = sd.InputStream(
                samplerate=actual_sample_rate,
                channels=1,
                device=self.device,
                callback=self._audio_callback,
                blocksize=1024  # Small blocksize for responsive level updates
            )
            self.stream.start()

            logger.debug("Audio monitor started")

        except Exception as e:
            logger.error(f"Failed to start audio monitor: {e}")
            self.is_monitoring = False
            raise

    def stop(self):
        """Stop monitoring audio levels."""
        if not self.is_monitoring:
            return

        logger.debug("Stopping audio monitor...")
        self.is_monitoring = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        with self._lock:
            self.current_level = 0.0

        logger.debug("Audio monitor stopped")

    def get_level(self) -> float:
        """
        Get current audio level.

        Returns:
            Audio level as percentage (0-100)
        """
        with self._lock:
            return self.current_level

    @staticmethod
    def list_input_devices():
        """
        List all available input audio devices.

        Returns:
            List of dicts with device info
        """
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': int(device['default_samplerate'])
                })

        return input_devices


def test_monitor():
    """Test audio device monitor."""
    logger.info("Testing Audio Device Monitor...")

    # List devices
    devices = AudioDeviceMonitor.list_input_devices()
    logger.info(f"Found {len(devices)} input devices:")
    for dev in devices:
        logger.info(f"  [{dev['index']}] {dev['name']} - {dev['channels']}ch @ {dev['sample_rate']}Hz")

    if not devices:
        logger.error("No input devices found!")
        return

    # Test first device
    device_idx = devices[0]['index']
    logger.info(f"\nTesting device {device_idx}: {devices[0]['name']}")

    monitor = AudioDeviceMonitor(device=device_idx)
    monitor.start()

    logger.info("Monitoring audio level for 10 seconds (speak into mic)...")
    for i in range(10):
        time.sleep(1)
        level = monitor.get_level()
        bar = '█' * int(level / 2)  # 50 char max
        logger.info(f"Level: {level:5.1f}% |{bar:<50}|")

    monitor.stop()
    logger.info("Test complete!")


if __name__ == "__main__":
    import sys
    import os

    # Setup logging
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/device_monitor_test.log")

    test_monitor()
