"""Shared helpers: seeding, config, Pashto text handling, metrics, run metadata.

Reproduction of exp_009 from H. Rahman, "Fine-tuning Whisper for Pashto ASR:
strategies and scale", arXiv:2604.06507 (Apr. 2026).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches. Paper used seed 42.

    Note for the report: this makes runs reproducible on the SAME hardware.
    It does not give bitwise equality across different GPU models, because
    cuDNN kernel selection and fp16 accumulation order are device-dependent.
    """
    import torch
    import transformers

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_effective_config(cfg: dict[str, Any], run_dir: str | Path) -> Path:
    """Write the config actually used, after CLI overrides. Required evidence."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "effective_config.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return out


def write_run_meta(run_dir: str | Path, **extra: Any) -> Path:
    """git SHA, GPU, wall clock, peak VRAM -- traceability for every number."""
    import torch

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sha = "unknown"

    meta = {
        "git_commit": sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count(),
        "peak_vram_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available() else None
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **extra,
    }
    out = run_dir / "run_meta.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out


# --------------------------------------------------------------------------
# Pashto text handling
# --------------------------------------------------------------------------

# Characters that identify genuine Pashto orthography. Used to (a) reject rows
# with no Arabic-script content and (b) measure the "script failure" rate that
# arXiv:2604.04598 reports as its headline zero-shot finding (< 0.8% of
# utterances contain Pashto-script output).
PASHTO_DISTINCTIVE = set("ټډړڼږښځڅګۍئ")
ARABIC_SCRIPT_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

# Ablation E3b only. The paper applies NO Pashto-specific normalisation --
# it uses HF BasicTextNormalizer at metric time, which performs no
# Arabic-script canonicalisation. Keep this OFF for the main run.
PASHTO_CANONICAL_MAP = {
    "ي": "ی",  # Arabic yeh      -> Pashto yeh
    "ى": "ی",  # alef maksura    -> Pashto yeh
    "ك": "ک",  # Arabic kaf      -> Pashto kaf
    "ۀ": "ه",  # heh with yeh above -> heh
    "ة": "ه",  # teh marbuta     -> heh
}
_STRIP_CHARS = (
    "‌‍​"          # ZWNJ, ZWJ, ZWSP
    "ـ"                      # tatweel / kashida
    "ًٌٍَُِّْٰ"  # harakat
    "،؛؟۔"    # Arabic comma, semicolon, question mark, full stop
)
_DIGIT_MAP = {**{chr(0x0660 + i): str(i) for i in range(10)},
              **{chr(0x06F0 + i): str(i) for i in range(10)}}


def has_arabic_script(text: str) -> bool:
    return any(
        any(lo <= ord(ch) <= hi for lo, hi in ARABIC_SCRIPT_RANGES) for ch in text
    )


def pashto_script_ratio(text: str) -> float:
    """Fraction of characters that are Pashto-distinctive. 0.0 for Arabic/Urdu output."""
    if not text:
        return 0.0
    return sum(ch in PASHTO_DISTINCTIVE for ch in text) / len(text)


def normalise_pashto(text: str) -> str:
    """Ablation E3b. Never alter the distinctive letters -- that destroys the task."""
    for src, dst in PASHTO_CANONICAL_MAP.items():
        text = text.replace(src, dst)
    text = "".join(_DIGIT_MAP.get(ch, ch) for ch in text if ch not in _STRIP_CHARS)
    return " ".join(unicodedata.normalize("NFC", text).split())


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@dataclass
class MetricResult:
    wer_norm: float
    wer_ortho: float
    cer: float
    n_utterances: int
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    pashto_script_rate: float

    def as_row(self, **extra: Any) -> dict[str, Any]:
        total_err = self.substitutions + self.deletions + self.insertions
        pct = lambda n: round(100.0 * n / total_err, 2) if total_err else 0.0
        return {
            "wer": round(self.wer_norm, 4),
            "wer_ortho": round(self.wer_ortho, 4),
            "cer": round(self.cer, 4),
            "n_utterances": self.n_utterances,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "sub_pct_of_errors": pct(self.substitutions),
            "del_pct_of_errors": pct(self.deletions),
            "ins_pct_of_errors": pct(self.insertions),
            "pashto_script_rate": round(self.pashto_script_rate, 4),
            **extra,
        }


def compute_metrics(references: list[str], hypotheses: list[str]) -> MetricResult:
    """WER (normalised + orthographic), CER, and the S/D/I breakdown.

    Normalisation mirrors the paper: both sides through HF BasicTextNormalizer
    before scoring. The paper's headline metric is normalised WER, so the
    report's difference calculation must use `wer_norm` on both sides.
    """
    import evaluate as hf_evaluate
    import jiwer
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    assert len(references) == len(hypotheses), "reference/hypothesis length mismatch"

    normaliser = BasicTextNormalizer()
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    ref_n, hyp_n = [], []
    for ref, hyp in zip(references, hypotheses):
        r, h = normaliser(ref).strip(), normaliser(hyp).strip()
        if not r:            # empty reference makes WER undefined -- drop the pair
            continue
        ref_n.append(r)
        hyp_n.append(h if h else " ")

    wer_norm = 100.0 * wer_metric.compute(references=ref_n, predictions=hyp_n)
    cer = 100.0 * cer_metric.compute(references=ref_n, predictions=hyp_n)

    ortho_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    wer_ortho = 100.0 * wer_metric.compute(
        references=[r for r, _ in ortho_pairs],
        predictions=[h if h.strip() else " " for _, h in ortho_pairs],
    )

    out = jiwer.process_words(ref_n, hyp_n)
    script_rate = float(np.mean([pashto_script_ratio(h) > 0 for h in hypotheses])) if hypotheses else 0.0

    return MetricResult(
        wer_norm=wer_norm,
        wer_ortho=wer_ortho,
        cer=cer,
        n_utterances=len(ref_n),
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
        pashto_script_rate=script_rate,
    )


def utterance_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    """Per-utterance WER/CER and S/D/I, for the error breakdown and samples."""
    import jiwer

    if not reference.strip():
        return {"wer_utt": float("nan"), "cer_utt": float("nan"),
                "n_sub": 0, "n_del": 0, "n_ins": 0}
    hyp = hypothesis if hypothesis.strip() else " "
    w = jiwer.process_words(reference, hyp)
    c = jiwer.process_characters(reference, hyp)
    return {
        "wer_utt": round(100.0 * w.wer, 2),
        "cer_utt": round(100.0 * c.cer, 2),
        "n_sub": w.substitutions,
        "n_del": w.deletions,
        "n_ins": w.insertions,
    }


# --------------------------------------------------------------------------
# metrics.csv -- the Step 5 results table
# --------------------------------------------------------------------------

METRICS_COLUMNS = [
    "tag", "source", "experiment", "model", "split", "n_utterances",
    "wer", "wer_ortho", "cer", "paper_wer", "difference_pp", "difference_rel_pct",
    "substitutions", "deletions", "insertions", "hits",
    "sub_pct_of_errors", "del_pct_of_errors", "ins_pct_of_errors",
    "pashto_script_rate", "learning_rate", "effective_batch", "precision",
    "train_hours", "total_steps", "best_step", "best_epoch", "num_beams",
    "wall_clock_s", "gpu", "git_commit", "timestamp_utc", "notes",
]


def append_metrics_row(results_dir: str | Path, row: dict[str, Any]) -> Path:
    """Append one experiment to results/metrics.csv.

    `source` must be exactly "ours" or "paper" -- the figure scripts colour by
    it, so the paper/ours separation is visible in the charts, not just prose.
    """
    import pandas as pd

    assert row.get("source") in {"ours", "paper"}, "source must be 'ours' or 'paper'"

    if row.get("paper_wer") is not None and row.get("wer") is not None:
        diff = row["wer"] - row["paper_wer"]
        row["difference_pp"] = round(diff, 2)
        row["difference_rel_pct"] = round(100.0 * diff / row["paper_wer"], 2)

    path = Path(results_dir) / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{col: row.get(col) for col in METRICS_COLUMNS}])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
    return path
