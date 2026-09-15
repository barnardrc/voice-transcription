"""Configuration and ONNX embedding helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigLoader:
    @staticmethod
    def load(path: str | Path = "config.json") -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as config_file:
            return json.load(config_file)


class EmbeddingExtractor:
    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, audio_data):
        import numpy as np

        audio_float = audio_data.astype(np.float32)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0
        audio_tensor = np.expand_dims(audio_float, axis=0)
        return self.session.run(None, {self.input_name: audio_tensor})[0][0]
