# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:19:18 2026

@author: barna
"""
import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier

class SpeechBrainONNXWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model

    def forward(self, wavs):
        wav_lens = torch.ones(wavs.shape[0], device=wavs.device)
        feats = self.compute_features(wavs)
        feats = self.mean_var_norm(feats, wav_lens)
        embeddings = self.embedding_model(feats, wav_lens)
        return embeddings

def export_to_onnx(local_model_dir):
    classifier = EncoderClassifier.from_hparams(source=local_model_dir)
    wrapper = SpeechBrainONNXWrapper(classifier)
    wrapper.eval()
    
    dummy_input = torch.randn(1, 32000) 
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        "speaker_model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'audio_length'},
            'output': {0: 'batch_size'}
        }
    )

if __name__ == "__main__":
    export_to_onnx("./model_local")