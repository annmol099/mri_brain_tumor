"""Compatibility shim for the Streamlit app.

This re-exports the model architecture from `src.model_architecture` so the
existing import in `app_with_login.py` works without relying on runtime
path manipulation.
"""

from src.model_architecture import ResNet50Classifier

__all__ = ["ResNet50Classifier"]
