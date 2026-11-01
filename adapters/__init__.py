"""integrations package
Adapters translating raw model outputs from individual teammates/services
into the standardized ModalityResult interface.
"""

from . import numba_compat  # Ensure numba shim is active for librosa
from .audio_adapter import AudioAdapter
from .url_adapter import URLAdapter
from .text_adapter import TextAdapter
from .screenshot_adapter import ScreenshotAdapter

__all__ = [
    "AudioAdapter",
    "URLAdapter",
    "TextAdapter",
    "ScreenshotAdapter",
]
