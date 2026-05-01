"""
ml_engine — PyTorch CNN-LSTM Inference Engine
==============================================
Package providing the core machine-learning components for
Ultrasound Elastography Label-Free Tissue Tumor Grading.

Public API
----------
    from ml_engine import CNNLSTMElastography, load_model
    from ml_engine.signal_processing import process_rf_pipeline
    from ml_engine.inference import run_inference

Submodules
----------
    architecture      — PyTorch model classes (CNNLSTMElastography, UNetEncoder, ConvBlock)
    signal_processing — Hankelization, 2D-DFT filtering, Exact DMD, Rayleigh augmentation
    inference         — End-to-end inference helper (load .mat → predict → JSON response)
"""

from .architecture import (
    ConvBlock,
    UNetEncoder,
    CNNLSTMElastography,
)
from .inference import load_model, run_inference

__all__ = [
    "ConvBlock",
    "UNetEncoder",
    "CNNLSTMElastography",
    "load_model",
    "run_inference",
]

__version__ = "1.0.0"
