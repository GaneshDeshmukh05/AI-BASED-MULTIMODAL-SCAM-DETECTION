"""URL alias for model_services."""

import sys
from pathlib import Path

_cur = Path(__file__).resolve().parent
if str(_cur) not in sys.path:
    sys.path.insert(0, str(_cur))

from src.model_service import ModelService, get_model_service, ModelUnavailableError

__all__ = ["ModelService", "get_model_service", "ModelUnavailableError"]

if __name__ == "__main__":
    service = get_model_service()
    service.load()
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(f"Model Service loaded. Testing: {url}")
    print(service.analyze(url))
