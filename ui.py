# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:55:02 2026

@author: barna
"""
import gradio as gr
import numpy as np
from utils import ConfigLoader, EmbeddingExtractor

def generate_baseline_ui(audio):
    if audio is None:
        return "Please record or upload an audio file."
    
    sample_rate, audio_data = audio
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    config = ConfigLoader.load()
    extractor = EmbeddingExtractor(config['speaker_verification']['model_path'])
    embedding = extractor.extract(audio_data)
    
    output_path = config['speaker_verification']['baseline_embedding_path']
    np.save(output_path, embedding)
    
    return f"Baseline generated and saved to {output_path}."

interface = gr.Interface(
    fn=generate_baseline_ui,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Target User Audio"),
    outputs=gr.Textbox(label="Status"),
    title="Speaker Baseline Generator"
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)