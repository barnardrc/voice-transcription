# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:01:39 2026

@author: barna
"""

import json
import numpy as np
import onnxruntime as ort

class ConfigLoader:
    @staticmethod
    def load(path='config.json'):
        with open(path, 'r') as f:
            return json.load(f)

class EmbeddingExtractor:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, audio_data):
        audio_float = audio_data.astype(np.float32)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0
        audio_tensor = np.expand_dims(audio_float, axis=0)
        return self.session.run(None, {self.input_name: audio_tensor})[0][0]