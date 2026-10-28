"""Build the reproduction dataset from the official Common Voice v24 Pashto release.

Why we rebuild instead of loading the paper's dataset objects:
`ihanif/common-voice-pashto-24` and `ihanif/pashto_augmented_speech` both return
HTTP 401 and are absent from the author's public dataset list. The
`mozilla-foundation/common_voice_*` Hugging Face repos have been empty since
October 2025. So we derive the corpus from the official Mozilla Data Collective
archive using the shipped train/dev/test TSVs -- the same derivation the paper
describes. See docs/01 risk #1 and docs/03.

Outputs:
  data/processed/{train,dev,test}.jsonl
  results/preprocessing_log.csv     <- requirement (d): every step, with counts
  results/dataset_stats.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from utils import (
    dump_effective_config,
    has_arabic_script,
    load_config,
    normalise_pashto,
    set_all_seeds,
)

LOG_COLUMNS = [
    "step_id", "stage", "description", "rule", "param", "rows_in", "rows_out",
    "rows_removed", "pct_removed", "hours_in", "hours_out", "source",
    "deviation_from_paper", "timestamp",
]


class PreprocessingLog:
    """Emits results/preprocessing_log.csv. Never hand-write that file."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        self._next_id = 1

    def add(self, stage, description, rule, rows_in, rows_out, source,
            deviation="no", param="-", hours_in=None, hours_out=None):
        removed = rows_in - rows_out
        self.rows.append({
            "step_id": self._next_id,
            "stage": stage,
            "description": description,
            "rule": rule,
            "param": param,
            "rows_in": rows_in,
            "rows_out": rows_out,
            "rows_removed": removed,
            "pct_removed": round(100.0 * removed / rows_in, 4) if rows_in else 0.0,
            "hours_in": round(hours_in, 3) if hours_in is not None else "",
            "hours_out": round(hours_out, 3) if hours_out is not None else "",
            "source": source,
            "deviation_from_paper": deviation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self._next_id += 1
        print(f"  [{self._next_id - 1:>2}] {description}: {rows_in} -> {rows_out} "
              f"(-{removed})")

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"\nwrote {self.path}")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_durations(cv_root: Path) -> dict[str, float]:
    """clip_durations.tsv gives durations in ms -- avoids decoding every MP3."""
    path = cv_root / "clip_durations.tsv"
    if not path.exists():
        print("  clip_durations.tsv missing; durations probed with soundfile (slower)")
        return {}
    frame = pd.read_csv(path, sep="\t")
    name_col, ms_col = frame.columns[0], frame.columns[1]
    return dict(zip(frame[name_col], frame[ms_col] / 1000.0))


def probe_duration(path: Path) -> float:
    import soundfile as sf
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return float("nan")


def peak_amplitude(path: Path, sr: int) -> float:
    import librosa
    try:
        audio, _ = librosa.load(str(path), sr=sr, mono=True)
        return float(abs(audio).max()) if audio.size else 0.0
    except Exception:
        return 0.0


def prepare_split(split_name, tsv_name, cfg, cv_root, durations, log, tokenizer,
                  check_silence=False):
    data_cfg = cfg["data"]
    frame = pd.read_csv(cv_root / tsv_name, sep="\t", quoting=csv.QUOTE_NONE,
                        dtype=str, keep_default_na=False)
    n0 = len(frame)
    log.add("load", f"Load official {tsv_name} ({split_name})",
            "official shipped TSV, validated clips only", n0, n0,
            "paper Sec 2.2 / Common Voice official split")

    # -- Stage 1: text filtering -------------------------------------------
    n = len(frame)
    frame = frame[frame["sentence"].str.strip() != ""]
    log.add("filter", f"[{split_name}] Drop empty transcripts",
            'sentence.strip() == ""', n, len(frame), "our addition",
            deviation="yes - paper silent")

    if data_cfg["drop_non_arabic_script"]:
        n = len(frame)
        frame = frame[frame["sentence"].map(has_arabic_script)]
        log.add("filter", f"[{split_name}] Drop rows with no Arabic-script content",
                "no codepoint in Arabic blocks", n, len(frame), "our addition",
                deviation="yes - paper silent")

    n = len(frame)
    frame = frame.drop_duplicates(subset=data_cfg["dedupe_on"], keep="first")
    log.add("filter", f"[{split_name}] Deduplicate",
            f"drop duplicate {tuple(data_cfg['dedupe_on'])}", n, len(frame),
            "our addition", deviation="yes - paper silent")

    n = len(frame)
    lengths = [len(tokenizer(s).input_ids) for s in frame["sentence"]]
    frame = frame[[ln <= data_cfg["max_label_tokens"] for ln in lengths]]
    log.add("filter", f"[{split_name}] Drop over-long labels",
            "tokenized length > limit", n, len(frame),
            "Whisper decoder limit", param=data_cfg["max_label_tokens"])

    # -- Stage 2: audio ----------------------------------------------------
    frame["duration_s"] = [
        durations.get(p) if durations.get(p) is not None
        else probe_duration(cv_root / "clips" / p)
        for p in frame["path"]
    ]
    hours_in = frame["duration_s"].sum() / 3600.0

    n = len(frame)
    frame = frame[frame["duration_s"].between(
        data_cfg["min_duration_s"], data_cfg["max_duration_s"], inclusive="both")]
    log.add("filter", f"[{split_name}] Duration window",
            f"{data_cfg['min_duration_s']}s <= dur <= {data_cfg['max_duration_s']}s",
            n, len(frame), "Whisper 30s window; paper CV20 floor was 1.25s",
            param=f"{data_cfg['min_duration_s']}-{data_cfg['max_duration_s']}s",
            hours_in=hours_in, hours_out=frame["duration_s"].sum() / 3600.0)

    if check_silence:
        n = len(frame)
        keep = [peak_amplitude(cv_root / "clips" / p, data_cfg["sampling_rate"])
                >= data_cfg["min_peak_amplitude"] for p in frame["path"]]
        frame = frame[keep]
        log.add("filter", f"[{split_name}] Drop silent / near-silent clips",
                "peak amplitude < threshold", n, len(frame), "our addition",
                deviation="yes - paper silent", param=data_cfg["min_peak_amplitude"])

    log.add("audio", f"[{split_name}] Resample to 16 kHz mono",
            "decode MP3 -> 16000 Hz, 1 channel (applied at load time)",
            len(frame), len(frame), "paper Sec 2.2",
            param=data_cfg["sampling_rate"])

    # -- Stage 3: text normalisation ---------------------------------------
    if data_cfg["pashto_normalise_labels"]:
        frame["sentence"] = frame["sentence"].map(normalise_pashto)
        log.add("normalise", f"[{split_name}] Pashto canonicalisation of LABELS",
                "yeh/kaf/heh unify, strip ZWNJ+tatweel+harakat, digits -> ASCII, NFC",
                len(frame), len(frame), "our ablation E3b",
                deviation="yes - ABLATION ONLY, not the main run")
    else:
        log.add("normalise", f"[{split_name}] Labels kept RAW",
                "no label normalisation; BasicTextNormalizer applied at metric time only",
                len(frame), len(frame), "paper: metrics use BasicTextNormalizer")

    return frame.reset_index(drop=True)


def subset_by_hours(frame, hours, seed):
    """Seeded subset WITHIN a split. Never sample across split boundaries."""
    if hours is None or frame["duration_s"].sum() / 3600.0 <= hours:
        return frame
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cumulative = shuffled["duration_s"].cumsum() / 3600.0
    return shuffled[cumulative <= hours].reset_index(drop=True)


def split_stats(frame):
    durations = frame["duration_s"]
    return {
        "rows": int(len(frame)),
        "hours": round(float(durations.sum()) / 3600.0, 3),
        "speakers": int(frame["client_id"].nunique()),
        "duration_stats_s": {
            "min": round(float(durations.min()), 2),
            "p50": round(float(durations.quantile(0.50)), 2),
            "mean": round(float(durations.mean()), 2),
            "p95": round(float(durations.quantile(0.95)), 2),
            "max": round(float(durations.max()), 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cv-root", default=None)
    parser.add_argument("--train-hours", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--archive", default=None,
                        help="path to the downloaded archive, for the SHA-256 record")
    parser.add_argument("--check-silence", action="store_true",
                        help="decode every clip to test for silence (slow)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cv_root:
        cfg["paths"]["cv_root"] = args.cv_root
    if args.train_hours is not None:
        cfg["data"]["train_hours"] = args.train_hours
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = cfg["seed"]
    set_all_seeds(seed)

    cv_root = Path(cfg["paths"]["cv_root"])
    out_dir = Path(cfg["paths"]["processed"])
    results_dir = Path(cfg["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in (cfg["data"]["train_tsv"], cfg["data"]["dev_tsv"], cfg["data"]["test_tsv"]):
        if not (cv_root / required).exists():
            raise SystemExit(
                f"missing {cv_root / required}\n"
                "Download Common Voice Scripted Speech 24.0, locale 'ps', from\n"
                "  https://datacollective.mozillafoundation.org/datasets/"
                "cmj8u3pnb00llnxxbfvxo3b14\n"
                "and extract so that <cv_root>/{clips,train.tsv,dev.tsv,test.tsv} exists.\n"
                "Note: the mozilla-foundation HF repos have been EMPTY since Oct 2025."
            )

    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg["model"]["name"], language=cfg["model"]["language"], task=cfg["model"]["task"]
    )

    log = PreprocessingLog(results_dir / "preprocessing_log.csv")
    durations = load_durations(cv_root)

    print("\n== train ==")
    train = prepare_split("train", cfg["data"]["train_tsv"], cfg, cv_root,
                          durations, log, tokenizer, args.check_silence)
    print("\n== dev ==")
    dev = prepare_split("dev", cfg["data"]["dev_tsv"], cfg, cv_root,
                        durations, log, tokenizer, args.check_silence)
    print("\n== test ==")
    test = prepare_split("test", cfg["data"]["test_tsv"], cfg, cv_root,
                         durations, log, tokenizer, args.check_silence)

    # ---- official-split integrity assertions (docs/03 Sec 3.7) -----------
    speakers = {"train": set(train["client_id"]), "dev": set(dev["client_id"]),
                "test": set(test["client_id"])}
    disjoint = {
        "train_dev": not (speakers["train"] & speakers["dev"]),
        "train_test": not (speakers["train"] & speakers["test"]),
        "dev_test": not (speakers["dev"] & speakers["test"]),
    }
    for pair, ok in disjoint.items():
        print(f"speaker-disjoint {pair}: {ok}")
    # Risk #6: the paper reports dev and test both at 13,791 rows.
    overlap = len(set(dev["sentence"]) & set(test["sentence"]))
    print(f"dev/test identical-sentence overlap: {overlap}")

    train_full = split_stats(train)
    train = subset_by_hours(train, cfg["data"]["train_hours"], seed)
    log.add("subset", "[train] Seeded subset within the train split",
            "shuffle(seed) then take clips until the hour budget is met",
            train_full["rows"], len(train),
            "our change - compute limit (paper used 113.4 h train+dev)",
            deviation="yes - LARGEST source of the WER gap",
            param=f"{cfg['data']['train_hours']} h, seed={seed}",
            hours_in=train_full["hours"], hours_out=train["duration_s"].sum() / 3600.0)

    dev_full_rows = len(dev)
    if cfg["data"]["dev_subset_rows"]:
        dev = dev.sample(n=min(cfg["data"]["dev_subset_rows"], len(dev)),
                         random_state=seed).reset_index(drop=True)
    log.add("subset", "[dev] Seeded dev subsample, held OUT of training",
            "random sample(seed)", dev_full_rows, len(dev),
            "our change - paper merged dev into train, leaving no clean val set",
            deviation="yes - makes checkpoint selection honest; costs us ~15% of rows",
            param=f"{cfg['data']['dev_subset_rows']} rows, seed={seed}")

    if cfg["data"]["test_rows"]:
        n = len(test)
        test = test.sample(n=min(cfg["data"]["test_rows"], len(test)),
                           random_state=seed).reset_index(drop=True)
        log.add("subset", "[test] Test subsample", "random sample(seed)", n, len(test),
                "our change - time budget",
                deviation="yes - REDUCES comparability with the paper's N=13,791")

    for name, frame in (("train", train), ("dev", dev), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in frame.itertuples(index=False):
                fh.write(json.dumps({
                    "audio": str((cv_root / "clips" / row.path).as_posix()),
                    "sentence": row.sentence,
                    "client_id": row.client_id,
                    "duration_s": round(float(row.duration_s), 3),
                }, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(frame)} rows)")

    stats = {
        "cv_version": cfg["data"]["cv_version"],
        "locale": cfg["data"]["locale"],
        "archive_sha256": sha256(Path(args.archive)) if args.archive else None,
        "download_date": time.strftime("%Y-%m-%d"),
        "seed": seed,
        "splits": {
            "train_full": train_full,
            "train_subset": split_stats(train),
            "dev_subset": split_stats(dev),
            "test": split_stats(test),
        },
        "speaker_disjoint": disjoint,
        "dev_test_sentence_overlap": overlap,
        "paper_reference": cfg["paper_reference"]["splits"],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"wrote {results_dir / 'dataset_stats.json'}")

    log.write()
    dump_effective_config(cfg, results_dir / "prepare_data")

    print("\n-- summary --")
    print(f"train  {stats['splits']['train_subset']['rows']:>7} rows  "
          f"{stats['splits']['train_subset']['hours']:>8.2f} h   (paper: 92,880 / 113.40 h)")
    print(f"dev    {stats['splits']['dev_subset']['rows']:>7} rows  "
          f"{stats['splits']['dev_subset']['hours']:>8.2f} h   (paper: merged into train)")
    print(f"test   {stats['splits']['test']['rows']:>7} rows  "
          f"{stats['splits']['test']['hours']:>8.2f} h   (paper: 13,791 / 20.70 h)")


if __name__ == "__main__":
    main()
