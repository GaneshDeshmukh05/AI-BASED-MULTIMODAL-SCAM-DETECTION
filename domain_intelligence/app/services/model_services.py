"""app/services/model_services.py alias for model_service."""

from app.services.model_service import ModelService, get_model_service, ModelUnavailableError

__all__ = ["ModelService", "get_model_service", "ModelUnavailableError"]
