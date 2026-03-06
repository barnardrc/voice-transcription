# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:22:17 2026

@author: barna
"""

import queue
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from pywhispercpp.model import Model
from scipy.spatial.distance import cosine
from utils import ConfigLoader, EmbeddingExtractor

class VADProcessor:
    def __init__(self, model_path, threshold, sample_rate):
        self.session = ort.InferenceSession(model_path)
        self.threshold = threshold
        self.sample_rate = np.array([sample_rate], dtype=np.int64)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_bytes):
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = (audio_data / 32768.0).astype(np.float32)
        audio_tensor = np.expand_dims(audio_float, axis=0)
        
        ort_inputs = {
            'input': audio_tensor,
            'sr': self.sample_rate,
            'state': self.state
        }
        
        out, self.state = self.session.run(None, ort_inputs)
        return out[0][0] >= self.threshold

class SpeakerRecognizer:
    def __init__(self, extractor, baseline_path, threshold):
        self.extractor = extractor
        self.baseline = np.load(baseline_path)
        self.threshold = threshold

    def verify(self, audio_buffer):
        audio_data = np.concatenate(audio_buffer).flatten()
        embedding = self.extractor.extract(audio_data)
        similarity = 1 - cosine(self.baseline, embedding)
        
        is_target = similarity >= self.threshold
        return is_target, similarity

class Transcriber:
    def __init__(self, model_name, sample_rate):
        # pywhispercpp will automatically download the model (e.g., 'tiny.en') if it is not found locally.
        self.model = Model(model_name, n_threads=4, print_realtime=False, print_progress=False)

    def transcribe(self, audio_buffer):
        audio_data = b''.join(audio_buffer)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # pywhispercpp expects audio as a float32 numpy array normalized between -1.0 and 1.0
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        segments = self.model.transcribe(audio_float)
        return " ".join([segment.text for segment in segments]).strip()

class AudioStreamer:
    def __init__(self, config):
        self.config = config
        self.queue = queue.Queue()
        self.vad = VADProcessor(
            config['vad']['model_path'],
            config['vad']['threshold'],
            config['audio']['sample_rate']
        )
        extractor = EmbeddingExtractor(config['speaker_verification']['model_path'])
        self.recognizer = SpeakerRecognizer(
            extractor,
            config['speaker_verification']['baseline_embedding_path'],
            config['speaker_verification']['threshold']
        )
        self.transcriber = Transcriber(
            config['stt']['model_path'],
            config['audio']['sample_rate']
        )

    def audio_callback(self, indata, frames, time, status):
        self.queue.put(indata.copy())

    def run(self):
        sample_rate = self.config['audio']['sample_rate']
        frame_size = int(sample_rate * (self.config['audio']['frame_duration_ms'] / 1000.0))
        min_frames = self.config['vad']['min_frames_for_processing']

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=self.config['audio']['channels'],
            dtype='int16',
            blocksize=frame_size,
            callback=self.audio_callback
        )
        
        speech_buffer = []
        is_speaking = False

        with stream:
            while True:
                chunk = self.queue.get()
                if self.vad.is_speech(chunk.tobytes()):
                    speech_buffer.append(chunk)
                    is_speaking = True
                elif is_speaking:
                    if len(speech_buffer) >= min_frames:
                        is_target, score = self.recognizer.verify(speech_buffer)
                        if is_target:
                            text = self.transcriber.transcribe(speech_buffer)
                            print(f"[Target - Score: {score:.2f}]: {text}")
                        else:
                            print(f"[Ignored - Score: {score:.2f}]")
                    speech_buffer = []
                    is_speaking = False

if __name__ == '__main__':
    config = ConfigLoader.load()
    app = AudioStreamer(config)
    app.run()