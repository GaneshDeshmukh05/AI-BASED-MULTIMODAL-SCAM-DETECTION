"""numpy_cnn.py
High-performance pure-NumPy forward-pass inference engine for 2D CNN models.
Allows running trained scam and anti-spoof Keras models (.keras) without
requiring TensorFlow or C-extensions, fully bypassing Windows Application Control
and Smart App Control policy blocks.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any, List, Optional

import h5py
import numpy as np

logger = logging.getLogger("numpy_cnn")


def _read_array(dataset: Any) -> np.ndarray:
    """Materialize an HDF5 dataset node as a float32 NumPy array."""
    return np.asarray(dataset[...], dtype=np.float32)


class NumPyKerasCNNPredictor:
    """Pure-NumPy forward pass evaluator for Keras 3 CNN architectures."""

    def __init__(self, keras_model_path: str | Path, has_conv_bias: bool = False):
        self.keras_model_path = str(keras_model_path)
        self.has_conv_bias = has_conv_bias
        self._loaded = False
        self._load_weights()

    def _load_weights(self) -> None:
        if not Path(self.keras_model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.keras_model_path}")

        with zipfile.ZipFile(self.keras_model_path, "r") as zf:
            if "model.weights.h5" not in zf.namelist():
                raise ValueError(f"Invalid .keras archive (missing model.weights.h5): {self.keras_model_path}")

            h5_bytes = zf.read("model.weights.h5")
            with h5py.File(io.BytesIO(h5_bytes), "r") as hf:
                layers: Any = hf["layers"]
                # Conv Block 1
                self.w1 = _read_array(layers["conv2d"]["vars"]["0"])
                self.b1 = (
                    _read_array(layers["conv2d"]["vars"]["1"])
                    if self.has_conv_bias and "1" in layers["conv2d"]["vars"]
                    else None
                )
                self.bn1 = [
                    _read_array(layers["batch_normalization"]["vars"][str(i)])
                    for i in range(4)
                ]

                # Conv Block 2
                self.w2 = _read_array(layers["conv2d_1"]["vars"]["0"])
                self.b2 = (
                    _read_array(layers["conv2d_1"]["vars"]["1"])
                    if self.has_conv_bias and "1" in layers["conv2d_1"]["vars"]
                    else None
                )
                self.bn2 = [
                    _read_array(layers["batch_normalization_1"]["vars"][str(i)])
                    for i in range(4)
                ]

                # Conv Block 3
                self.w3 = _read_array(layers["conv2d_2"]["vars"]["0"])
                self.b3 = (
                    _read_array(layers["conv2d_2"]["vars"]["1"])
                    if self.has_conv_bias and "1" in layers["conv2d_2"]["vars"]
                    else None
                )
                self.bn3 = [
                    _read_array(layers["batch_normalization_2"]["vars"][str(i)])
                    for i in range(4)
                ]

                # Dense Block
                self.wd1 = _read_array(layers["dense"]["vars"]["0"])
                self.bd1 = _read_array(layers["dense"]["vars"]["1"])
                self.wd2 = _read_array(layers["dense_1"]["vars"]["0"])
                self.bd2 = _read_array(layers["dense_1"]["vars"]["1"])

        self._loaded = True

    @staticmethod
    def _conv2d(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
        """2D convolution with padding='same' (3x3 kernel, stride=1)."""
        padded = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))
        out = np.tensordot(windows, w, axes=([3, 4, 2], [0, 1, 2]))
        if b is not None:
            out += b
        return out

    @staticmethod
    def _batchnorm(x: np.ndarray, params: List[np.ndarray], eps: float = 0.001) -> np.ndarray:
        """Batch normalization: (x - mean) * (gamma / sqrt(var + eps)) + beta."""
        gamma, beta, mean, var = params
        inv_std = 1.0 / np.sqrt(var + eps)
        return (x - mean) * (inv_std * gamma) + beta

    @staticmethod
    def _maxpool2d(x: np.ndarray) -> np.ndarray:
        """2x2 max pooling with stride 2."""
        H, W, C = x.shape
        H_out = H // 2
        W_out = W // 2
        cropped = x[: H_out * 2, : W_out * 2, :]
        return cropped.reshape(H_out, 2, W_out, 2, C).max(axis=(1, 3))

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Executes forward pass inference matching Keras Model.predict interface."""
        # Normalize batch dimension
        if x.ndim == 4:
            x_sample = x[0]
        elif x.ndim == 3:
            x_sample = x
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

        # Block 1
        x1 = self._conv2d(x_sample, self.w1, self.b1)
        x1 = np.maximum(0, self._batchnorm(x1, self.bn1))
        x1 = self._maxpool2d(x1)

        # Block 2
        x2 = self._conv2d(x1, self.w2, self.b2)
        x2 = np.maximum(0, self._batchnorm(x2, self.bn2))
        x2 = self._maxpool2d(x2)

        # Block 3
        x3 = self._conv2d(x2, self.w3, self.b3)
        x3 = np.maximum(0, self._batchnorm(x3, self.bn3))

        # Global Average Pooling (across H and W dimensions)
        gap = x3.mean(axis=(0, 1))

        # Dense 1 + ReLU
        d1 = np.maximum(0, np.dot(gap, self.wd1) + self.bd1)

        # Dense 2 + Sigmoid
        z = np.dot(d1, self.wd2) + self.bd2
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

        return np.array([[float(prob.ravel()[0])]], dtype=np.float32)
