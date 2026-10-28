"""Single-file (or folder) Pashto transcription -- the demo for Appendix F.

    python src/infer.py --model results/exp009_repro_tierB/best --audio clip.mp3
    python src/infer.py --model ihanif/exp_009_small_cv24_vanilla --audio data/samples/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils import load_config, pashto_script_ratio, set_all_seeds

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True, help="file or directory")
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--out", default=None, help="optional CSV output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])

    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    target = Path(args.audio)
    files = (
        sorted(p for p in target.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        if target.is_dir() else [target]
    )
    if not files:
        raise SystemExit(f"no audio found at {target}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    sr = cfg["data"]["sampling_rate"]

    processor = WhisperProcessor.from_pretrained(
        cfg["model"]["name"], language=cfg["model"]["language"], task=cfg["model"]["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.eval()

    num_beams = args.num_beams or cfg["train"]["generation_num_beams"]
    rows = []

    for path in files:
        audio, _ = librosa.load(str(path), sr=sr, mono=True)  # 16 kHz mono
        features = processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(device)
        if dtype == torch.float16:
            features = features.half()

        with torch.no_grad():
            generated = model.generate(
                features,
                num_beams=num_beams,
                max_new_tokens=cfg["train"]["generation_max_length"],
                language=cfg["model"]["language"],
                task=cfg["model"]["task"],
            )
        text = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        rows.append({
            "file": path.name,
            "duration_s": round(len(audio) / sr, 2),
            "transcript": text,
            # > 0 means genuine Pashto orthography; 0 usually means the model
            # fell back to Arabic/Dari/Urdu script -- the failure mode
            # arXiv:2604.04598 quantifies for zero-shot Whisper.
            "pashto_script_ratio": round(pashto_script_ratio(text), 4),
        })
        print(f"\n{path.name}  ({rows[-1]['duration_s']}s)")
        print(f"  {text}")

    if args.out:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
