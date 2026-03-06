# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:39:50 2026

@author: barna
"""

import numpy as np
import soundfile as sf
from utils import ConfigLoader, EmbeddingExtractor

def generate_baseline(audio_path, config_path='config.json'):
    config = ConfigLoader.load(config_path)
    extractor = EmbeddingExtractor(config['speaker_verification']['model_path'])
    
    audio_data, _ = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
        
    embedding = extractor.extract(audio_data)
    np.save(config['speaker_verification']['baseline_embedding_path'], embedding)

if __name__ == '__main__':
    generate_baseline('target_user_sample.wav')