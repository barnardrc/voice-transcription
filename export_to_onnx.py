"""Download the documented SpeechBrain model and export its embedder to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier


DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class SpeechBrainONNXWrapper(nn.Module):
    def __init__(self, classifier: EncoderClassifier) -> None:
        super().__init__()
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        wav_lens = torch.ones(wavs.shape[0], device=wavs.device)
        features = self.compute_features(wavs)
        features = self.mean_var_norm(features, wav_lens)
        return self.embedding_model(features, wav_lens)


def export_to_onnx(
    *,
    source: str = DEFAULT_SOURCE,
    cache_dir: Path = Path("model_cache/spkrec-ecapa-voxceleb"),
    output_path: Path = Path("speaker_model.onnx"),
) -> None:
    classifier = EncoderClassifier.from_hparams(
        source=source,
        savedir=str(cache_dir),
    )
    wrapper = SpeechBrainONNXWrapper(classifier)
    wrapper.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 32_000)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "audio_length"},
            "output": {0: "batch_size"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("model_cache/spkrec-ecapa-voxceleb")
    )
    parser.add_argument("--output", type=Path, default=Path("speaker_model.onnx"))
    args = parser.parse_args()
    export_to_onnx(source=args.source, cache_dir=args.cache_dir, output_path=args.output)


if __name__ == "__main__":
    main()
