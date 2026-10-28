# Fine-Tuning Whisper for Pashto ASR — Reproduction

Reproduction of **exp_009** from:

> H. Rahman, "Fine-tuning Whisper for Pashto ASR: strategies and scale,"
> arXiv:2604.06507, Apr. 2026. doi: [10.48550/arXiv.2604.06507](https://doi.org/10.48550/arXiv.2604.06507)

**Target to reproduce:** `whisper-small`, full fine-tuning, Common Voice Pashto v24 →
**WER 24.89%**, **CER 8.075%**, best checkpoint step 8500 (epoch 5.85).
**Paired zero-shot baseline:** 120.4% WER.

CS-444 Deep Learning · Assignment #02 · Air University Islamabad · Section BSCS-VII-B (Shift-II) · Instructor: Mr. Qadeer Yasin

---

## Status

| | |
|---|---|
| Base paper validated | ✅ `docs/01_paper_validation.md` |
| Dataset located | ✅ Mozilla Data Collective (HF mirrors are empty / 401) |
| Author's checkpoint located | ✅ `ihanif/exp_009_small_cv24_vanilla`, Apache-2.0 |
| Author's training code | ❌ not public → method re-implemented here |
| Runs executed | ⬜ not yet — all result cells are `TBD` by design |

## Read the plan before running anything

| Doc | Covers | Assignment requirement |
|---|---|---|
| [`docs/01_paper_validation.md`](docs/01_paper_validation.md) | Verified paper record, reported results, the exact experiment, 13 reproduction risks | (b) |
| [`docs/02_problem_and_model.md`](docs/02_problem_and_model.md) | Why ASR is prediction, Whisper architecture, transfer learning, loss, optimizer, WER/CER | (a), (c) |
| [`docs/03_dataset_and_preprocessing.md`](docs/03_dataset_and_preprocessing.md) | Dataset identity, subset tiers, split policy, 11-step pipeline, log format | (d) |
| [`docs/04_implementation_setup.md`](docs/04_implementation_setup.md) | Pinned environment, GPU needs, seeding, **the paper-vs-ours deviation table**, commands | (e) |
| [`docs/05_experiment_and_visualization.md`](docs/05_experiment_and_visualization.md) | E0–E3 run matrix, results-table template, 9 figures, gap decomposition | (f), (g) |
| [`docs/06_reproducibility_statement.md`](docs/06_reproducibility_statement.md) | Reproducibility section, ready to paste | (h) |
| [`docs/07_report_outline.md`](docs/07_report_outline.md) | 20-section PDF outline with page budgets | report |
| [`docs/08_references_ieee.md`](docs/08_references_ieee.md) | IEEE citations + integrity rules | references |
| [`docs/09_submission_checklist.md`](docs/09_submission_checklist.md) | Line-by-line submission checklist | submission |

## Quickstart

```bash
conda env create -f environment.yml && conda activate pashto-whisper
# or: pip install -r requirements.txt   (install torch separately, matched to your CUDA)
```

### Data

The `mozilla-foundation/common_voice_*` Hugging Face repos have been **empty since October 2025**, and the paper's own mirrors (`ihanif/common-voice-pashto-24`, `ihanif/pashto_augmented_speech`) return **HTTP 401**. Get the data from the official source instead:

1. Create a free account on [Mozilla Data Collective](https://datacollective.mozillafoundation.org/datasets/cmj8u3pnb00llnxxbfvxo3b14).
2. Download **Common Voice Scripted Speech 24.0**, locale **`ps`** (CC0-1.0).
3. Extract so that `data/raw/cv-corpus-24.0/ps/{clips,train.tsv,dev.tsv,test.tsv}` exists.
4. Record the archive SHA-256 — `prepare_data.py --archive <path>` does it for you.

On Kaggle you can attach the [CC0 mirror](https://www.kaggle.com/datasets/ataullahaali/common-voice-scripted-speech-24-0-pashto) instead; cite it **as a mirror**, not as the official source.

### Pipeline

```bash
# 1. preprocess -> data/processed/*.jsonl + results/preprocessing_log.csv + dataset_stats.json
python src/prepare_data.py --config config.yaml --cv-root data/raw/cv-corpus-24.0/ps --train-hours 12 --seed 42

# 2. E1 - zero-shot baseline (no training; paper target 120.4% WER)
python src/evaluate.py --config config.yaml --model openai/whisper-small --split test --tag zeroshot_small

# 3. E0 - validate our harness against the author's weights (paper target 24.89%)
python src/evaluate.py --config config.yaml --model ihanif/exp_009_small_cv24_vanilla --split test --tag author_checkpoint

# 4. E2 - train our reproduction
python src/train.py --config config.yaml --tag exp009_repro_tierB
#    resume after a Colab disconnect:
python src/train.py --config config.yaml --tag exp009_repro_tierB --resume-from-checkpoint auto

# 5. E2 - evaluate it
python src/evaluate.py --config config.yaml --model results/exp009_repro_tierB/best --split test --tag finetuned_small

# 6. figures + comparison table
python src/visualize.py --config config.yaml --tag exp009_repro_tierB

# 7. inference demo for the appendix
python src/infer.py --model results/exp009_repro_tierB/best --audio data/samples/
```

**Run E1 first.** It needs no training and immediately confirms the data is loading: if zero-shot WER is *not* above 100%, the problem is the data, not the model.

**Do not skip E0.** Evaluating the author's published checkpoint with our own harness separates *measurement* error from *training* error, and it is the single highest-value hour in the project. It is also what lets the report make two distinct claims instead of one vague one.

## Expected outcome, stated honestly up front

| Tier | Train data | Hardware | Expected WER |
|---|---|---|---|
| A smoke test | 0.5 h | any T4 | ~80–100% |
| **B primary** | **12 h** | **T4 16 GB** | **~35–45%** |
| C stretch | 25 h | P100 | ~30–38% |
| D faithful | 113 h | A100 40 GB | ~25% (paper: 24.89%) |

At Tier B we will **not** match 24.89%. A ~12–18 pp gap caused by using ~10% of the paper's training data is a correct, well-explained reproduction outcome, and `results/wer_vs_hours.png` quantifies it. Fabricating a match is the one thing that fails this assignment.

## Deviations from the paper

Full table with reasons and reproducibility impact in [`docs/04`](docs/04_implementation_setup.md). The five that matter:

| | Paper (exp_009) | Ours | Why |
|---|---|---|---|
| Training data | 113.4 h (train+dev) | ~12 h (train subset) | free-tier compute limit |
| Learning rate | 1e-4 | 1e-5 | 1e-4 overfits a 12 h subset; ablation E3a measures the cost |
| Precision | FP32 | fp16 | fp32 whisper-small exceeds 16 GB at a usable batch |
| Effective batch | 64 | 32 | VRAM |
| Training script | `fine_tune_whisper_enhanced.py` (not public) | re-implemented via `Seq2SeqTrainer` | no repository URL published |

Values the paper never states and we therefore **assume**: optimizer (`adamw_torch`), warmup (500 steps), weight decay (0.0), beam width (greedy), max generation length (225).

## Layout

```
pashto-whisper-asr-reproduction/
├── README.md               config.yaml           requirements.txt   environment.yml
├── docs/                   the nine planning documents above
├── data/                   raw/ (gitignored) + processed/ (gitignored)
├── notebooks/              Colab / Kaggle notebooks, committed WITH outputs
├── src/
│   ├── prepare_data.py     official TSV splits -> jsonl + preprocessing audit log
│   ├── train.py            full FT / LoRA / frozen-encoder, Seq2SeqTrainer
│   ├── evaluate.py         WER / WER-ortho / CER / S-D-I / script-failure rate
│   ├── infer.py            single-file transcription demo
│   ├── visualize.py        the nine figures + comparison table
│   └── utils.py            seeding, Pashto text handling, metrics, metrics.csv
├── results/                metrics.csv, preprocessing_log.csv, predictions_*.csv, figures
└── report/                 assignment2_report.pdf + evidence/
```

## Integrity

- No paper, author, DOI, URL, dataset, hyperparameter or result value appears anywhere unless it was read from the source. Cells awaiting a run stay `TBD`.
- `results/metrics.csv` carries a `source` column with exactly two values, `paper` and `ours`; the figure scripts colour by it, so the separation is visible in the charts.
- Paper values live only in `config.yaml` → `paper_reference`, verified against arXiv:2604.06507 on 2026-10-27.
- `ihanif/exp_009_small_cv24_vanilla` is the author's work (Apache-2.0). We *evaluate* it; it is never presented as our training result.
- Common Voice v24 is CC0-1.0. We cite the source and do not redistribute the archive.

## Two gotchas that cost days

1. **Pashto rendering in the PDF.** `pdflatex` will not render Arabic script. Use XeLaTeX/LuaLaTeX with `fontspec` + `polyglossia` and Noto Naskh Arabic, then open the exported PDF on a second machine before submitting.
2. **Colab disconnects.** Checkpoint every 500 steps, mirror `results/<tag>/best/` to Google Drive, and always resume with `--resume-from-checkpoint auto`.

## Licence

Our code: MIT. Common Voice v24: CC0-1.0. `openai/whisper-small` and the author's checkpoints: Apache-2.0.
