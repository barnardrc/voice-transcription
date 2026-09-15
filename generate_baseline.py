"""Create a local speaker embedding from a consented audio sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from utils import ConfigLoader, EmbeddingExtractor


def generate_baseline(audio_path: Path, config_path: Path = Path("config.json")) -> Path:
    config = ConfigLoader.load(config_path)
    extractor = EmbeddingExtractor(config["speaker_verification"]["model_path"])

    audio_data, _ = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    embedding = extractor.extract(audio_data)
    output_path = Path(config["speaker_verification"]["baseline_embedding_path"])
    np.save(output_path, embedding)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path, help="Consented WAV or FLAC sample")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    output_path = generate_baseline(args.audio_path, args.config)
    print(f"Saved speaker baseline to {output_path}")


if __name__ == "__main__":
    main()
