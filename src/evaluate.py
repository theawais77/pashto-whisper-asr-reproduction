"""Evaluate any Whisper checkpoint on a Common Voice Pashto split.

Serves all four experiment roles from docs/05:
  E0  harness validation : --model ihanif/exp_009_small_cv24_vanilla
  E1  zero-shot baseline : --model openai/whisper-small
  E2  our reproduction   : --model results/<tag>/best
  E3  ablations          : --rescore-pashto-norm, --num-beams, ...

Appends one row to results/metrics.csv with source="ours" and the paper's
value in `paper_wer`, so the paper/ours separation is structural, not editorial.

Decoding settings MUST match between E1 and E2 or the comparison is void.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from utils import (
    append_metrics_row,
    compute_metrics,
    load_config,
    normalise_pashto,
    set_all_seeds,
    utterance_metrics,
)


def paper_reference_wer(cfg: dict, model_name: str, zero_shot: bool) -> tuple[float | None, str]:
    """Pick the comparable published value from arXiv:2604.06507.

    Only ever reads cfg["paper_reference"], which holds verified paper numbers
    and nothing of ours.
    """
    ref = cfg["paper_reference"]
    key = (
        "exp_010_turbo" if "turbo" in model_name or "large" in model_name
        else "exp_008_base" if "base" in model_name
        else "exp_009_small"
    )
    entry = ref[key]
    if zero_shot:
        return entry.get("zeroshot_wer"), f"{key} zero-shot"
    return entry.get("wer_norm"), key


def transcribe(model, processor, dataset, batch_size, num_beams, max_length, device):
    hypotheses, references, paths = [], [], []
    model.eval()

    for start in tqdm(range(0, len(dataset), batch_size), desc="decoding"):
        chunk = dataset[start:start + batch_size]
        features = processor.feature_extractor(
            chunk["audio_array"], sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        ).input_features.to(device)
        if model.dtype == torch.float16:
            features = features.half()

        with torch.no_grad():
            generated = model.generate(
                features, num_beams=num_beams, max_new_tokens=max_length,
                language=model.generation_config.language,
                task=model.generation_config.task,
            )
        hypotheses.extend(processor.tokenizer.batch_decode(generated, skip_special_tokens=True))
        references.extend(chunk["sentence"])
        paths.extend(chunk["audio_path"])

    return references, hypotheses, paths


def load_split(cfg: dict, split: str):
    from datasets import Audio, load_dataset

    path = Path(cfg["paths"]["processed"]) / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run src/prepare_data.py first")

    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.rename_column("audio", "audio_path")
    dataset = dataset.add_column("audio", list(dataset["audio_path"]))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=cfg["data"]["sampling_rate"]))
    return dataset.map(
        lambda row: {"audio_array": row["audio"]["array"]},
        remove_columns=["audio"], desc="loading audio",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True,
                        help="HF id or local path (e.g. results/<tag>/best)")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None,
                        help="repeat the run and report mean +/- std (paper-free noise estimate)")
    parser.add_argument("--rescore-pashto-norm", action="store_true",
                        help="ablation E3b: re-score with Pashto canonicalisation")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])

    num_beams = args.num_beams or cfg["train"]["generation_num_beams"]
    batch_size = args.batch_size or cfg["train"]["per_device_eval_batch_size"]
    repeats = args.repeats or cfg["eval"]["n_repeats"]
    max_length = cfg["train"]["generation_max_length"]

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

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

    # Zero-shot iff we are pointed at an untouched upstream checkpoint.
    zero_shot = args.model.startswith("openai/")
    paper_wer, paper_key = paper_reference_wer(cfg, args.model, zero_shot)

    dataset = load_split(cfg, args.split)
    print(f"model={args.model}  split={args.split}  N={len(dataset)}  "
          f"beams={num_beams}  zero_shot={zero_shot}")
    print(f"paper reference: {paper_key} = {paper_wer}% WER")

    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    wers, cers, last = [], [], None
    for run_index in range(repeats):
        start = time.time()
        references, hypotheses, paths = transcribe(
            model, processor, dataset, batch_size, num_beams, max_length, device
        )
        if args.rescore_pashto_norm:
            references = [normalise_pashto(r) for r in references]
            hypotheses = [normalise_pashto(h) for h in hypotheses]

        result = compute_metrics(references, hypotheses)
        wers.append(result.wer_norm)
        cers.append(result.cer)
        last = (result, references, hypotheses, paths, time.time() - start)
        print(f"  run {run_index + 1}/{repeats}: WER {result.wer_norm:.2f}  "
              f"CER {result.cer:.2f}  ({time.time() - start:.0f}s)")

    result, references, hypotheses, paths, wall_clock = last
    mean_wer = sum(wers) / len(wers)
    std_wer = (sum((w - mean_wer) ** 2 for w in wers) / len(wers)) ** 0.5

    rows = []
    for path, ref, hyp in zip(paths, references, hypotheses):
        rows.append({"audio_path": path, "reference": ref, "hypothesis": hyp,
                     **utterance_metrics(ref, hyp)})
    predictions = pd.DataFrame(rows)
    pred_path = results_dir / f"predictions_{args.tag}.csv"
    predictions.to_csv(pred_path, index=False, encoding="utf-8")

    # Sample from the best/median/worst terciles so the appendix table is
    # demonstrably not cherry-picked.
    n_sample = cfg["eval"]["sample_predictions_n"]
    ranked = predictions.dropna(subset=["wer_utt"]).sort_values("wer_utt")
    if len(ranked):
        per_bucket = max(1, n_sample // 3)
        mid = len(ranked) // 2
        sample = pd.concat([
            ranked.head(per_bucket).assign(bucket="best"),
            ranked.iloc[max(0, mid - per_bucket // 2): mid + per_bucket // 2 + 1].assign(bucket="median"),
            ranked.tail(per_bucket).assign(bucket="worst"),
        ])
        sample.insert(0, "tag", args.tag)
        sample_path = results_dir / "sample_predictions.csv"
        sample.to_csv(sample_path, mode="a", header=not sample_path.exists(),
                      index=False, encoding="utf-8")

    row = result.as_row(
        tag=args.tag,
        source="ours",
        experiment=("zero-shot" if zero_shot else
                    "author checkpoint" if args.model.startswith("ihanif/") else
                    "fine-tuned (ours)"),
        model=args.model,
        split=f"cv24-{args.split}",
        paper_wer=paper_wer,
        num_beams=num_beams,
        wall_clock_s=round(wall_clock, 1),
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        precision="fp16" if dtype == torch.float16 else "fp32",
        notes=(f"{args.notes} | mean+/-std over {repeats} runs: "
               f"{mean_wer:.2f}+/-{std_wer:.2f}"
               f"{' | E3b pashto-normalised rescoring' if args.rescore_pashto_norm else ''}"
               f" | paper ref: {paper_key}").strip(" |"),
    )
    append_metrics_row(results_dir, row)

    with open(results_dir / f"eval_{args.tag}.json", "w", encoding="utf-8") as fh:
        json.dump({**row, "wer_runs": wers, "cer_runs": cers,
                   "wer_mean": mean_wer, "wer_std": std_wer}, fh, indent=2)

    diff = result.wer_norm - paper_wer if paper_wer else None
    print(f"\n-- {args.tag} --")
    print(f"WER (normalised) {result.wer_norm:>7.2f} %   mean {mean_wer:.2f} +/- {std_wer:.2f}")
    print(f"WER (orthographic){result.wer_ortho:>6.2f} %")
    print(f"CER              {result.cer:>7.2f} %")
    print(f"paper ({paper_key}) {paper_wer:>6} %")
    if diff is not None:
        print(f"difference       {diff:>+7.2f} pp  "
              f"({100.0 * diff / paper_wer:+.1f} % relative)")
    print(f"errors: S {result.substitutions} / D {result.deletions} / "
          f"I {result.insertions}  (substitutions "
          f"{100.0 * result.substitutions / max(1, result.substitutions + result.deletions + result.insertions):.1f}% "
          f"of errors; paper reports ~70-78%)")
    print(f"Pashto-script output rate {100.0 * result.pashto_script_rate:.2f} % "
          f"(arXiv:2604.04598 reports <0.8% zero-shot)")
    print(f"\nwrote {pred_path}")


if __name__ == "__main__":
    main()
