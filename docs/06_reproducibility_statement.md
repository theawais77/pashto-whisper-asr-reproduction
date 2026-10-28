# Step 7 — Reproducibility Statement

Paste into report Section 17. Replace every `<...>` with a real value before submission. Keep the two-level verdict — it is the honest framing and it is also the strongest one.

---

## 17. Reproducibility Statement

### 17.1 Verdict

We separate two distinct claims, because they have different answers.

**(a) Is the original paper's reported result verifiable?** **Yes.** Using the author's published checkpoint `ihanif/exp_009_small_cv24_vanilla` and our own independent evaluation harness on the Common Voice v24 Pashto test split, we measured WER = `<E0 WER>`% and CER = `<E0 CER>`% against the paper's reported 24.89% / 8.075%. `<Agreement within X pp confirms / A discrepancy of X pp indicates ...>` This verifies the paper's *reported numbers* independently of its training pipeline.

**(b) Is the original paper's training procedure reproducible from the paper alone?** **Partially.** It is reproducible in *method* and *direction*, not in *exact value*, for three reasons that are properties of the paper, not of our effort:

1. The paper's training corpora (`ihanif/common-voice-pashto-24`, `ihanif/pashto_augmented_speech`) return HTTP 401 and are absent from the author's public dataset list. We rebuilt the corpus from the official Mozilla Data Collective v24 Pashto release using the shipped `train.tsv` / `dev.tsv` / `test.tsv` splits — the same derivation the author describes — but we cannot assert byte-level identity.
2. No public training or evaluation code exists. The paper names `fine_tune_whisper_enhanced.py` and an `experiments/scripts/` directory without a repository URL. We re-implemented the method with `Seq2SeqTrainer`.
3. The published configuration is incomplete. The **optimizer is never stated**; warmup, gradient-accumulation composition, beam width and maximum generation length are unstated for exp_009; and the text references a configuration "Table 4" that does not appear in the paper.

**(c) Is our own experiment reproducible from the materials we submit?** **Yes**, to within hardware-level numerical noise. Another student with a Mozilla Data Collective account and a 16 GB GPU can reproduce our reported WER following §17.4. We measured run-to-run variance over 3 evaluation runs at `<mean> ± <std>` pp.

### 17.2 What is included

| Item | Path | Status |
|---|---|---|
| Source code (data prep, training, evaluation, inference, figures) | `src/` | Included |
| Pinned pip requirements | `requirements.txt` | Included |
| Conda environment | `environment.yml` | Included |
| Full experiment configuration | `config.yaml` | Included |
| Resolved per-run configuration | `results/<tag>/effective_config.yaml` | Included |
| Captured environment (versions, GPU) | `results/environment.txt`, `results/pip_freeze.txt`, `results/gpu.txt` | Included |
| Random seed (42) and determinism flags | `src/utils.py::set_all_seeds` | Included |
| Preprocessing audit log, per rule, with row counts | `results/preprocessing_log.csv` | Included |
| Dataset statistics, split sizes, speaker-disjointness assertions, archive SHA-256 | `results/dataset_stats.json` | Included |
| Per-step training log (loss, eval loss, WER, CER, LR, grad-norm) | `results/<tag>/training_log.csv`, `trainer_state.json` | Included |
| TensorBoard event files | `results/<tag>/logs/` | Included |
| Aggregated metrics for every experiment, tagged `paper` vs `ours` | `results/metrics.csv` | Included |
| Full per-utterance predictions with per-utterance WER/CER and S/D/I counts | `results/predictions_<tag>.csv` | Included |
| Seeded sample predictions for the appendix | `results/sample_predictions.csv` | Included |
| All nine figures, PNG @300 dpi + PDF | `results/*.png`, `results/*.pdf` | Included |
| Run metadata (git SHA, wall clock, peak VRAM, GPU name) | `results/<tag>/run_meta.json` | Included |
| Colab/Kaggle notebook with cell outputs retained | `notebooks/` | Included |
| Execution evidence (timestamped screenshots, `nvidia-smi`, console logs) | `report/evidence/` | Included |
| Instructions to obtain the dataset | `README.md` §Data | Included |
| Instructions to obtain our checkpoint | `README.md` §Checkpoint | Included |

### 17.3 What is not included, and why

| Not included | Reason |
|---|---|
| The Common Voice v24 Pashto audio archive (~`<N>` GB) | License is CC0, so redistribution is legally permitted, but the archive is far too large for GCR and for a Git repository. We provide the official download URL, the archive SHA-256, and a preprocessing script that reproduces our exact subset from it given `seed=42`. |
| Derived `data/processed/` features | Regenerable in `<N>` minutes by `src/prepare_data.py`; tens of GB of log-Mel arrays. |
| Our fine-tuned model weights (~1 GB) | Exceeds GCR upload limits. `<Published at https://huggingface.co/<user>/<repo>, Apache-2.0 / Available on request>`. Intermediate checkpoints were pruned by `save_total_limit=2`. |
| The original paper's training corpus and scripts | Not public (HTTP 401 / no repository URL). Outside our control; documented as the paper's limitation, not ours. |
| Bit-identical reproduction across different GPU models | cuDNN kernel selection and fp16 accumulation order are device-dependent. Seeding fixes data order and initialisation, not kernel arithmetic. We report mean ± std over 3 runs so the magnitude of this effect is visible. |

### 17.4 Exact steps for another student to reproduce our result

```bash
# 1. Clone
git clone <REPO_URL> && cd pashto-whisper-asr-reproduction
git checkout <COMMIT_SHA>          # the exact commit used for the reported numbers

# 2. Environment (Python 3.10)
conda env create -f environment.yml && conda activate pashto-whisper
# or: pip install -r requirements.txt

# 3. Data — requires a free Mozilla Data Collective account
#    Download Common Voice Scripted Speech 24.0, locale "ps", from:
#    https://datacollective.mozillafoundation.org/datasets/cmj8u3pnb00llnxxbfvxo3b14
#    Extract to data/raw/ so that data/raw/cv-corpus-24.0/ps/{clips,train.tsv,dev.tsv,test.tsv} exists
sha256sum <archive>.tar.gz         # must equal <SHA256 in results/dataset_stats.json>

# 4. Preprocess (deterministic given seed 42)
python src/prepare_data.py --config config.yaml --cv-root data/raw/cv-corpus-24.0/ps --train-hours 12 --seed 42
#    Verify: results/preprocessing_log.csv and results/dataset_stats.json match ours

# 5. Zero-shot baseline  (~<T1>, no training)
python src/evaluate.py --config config.yaml --model openai/whisper-small --split test --tag zeroshot_small

# 6. Validate the harness against the author's checkpoint (~<T2>)
python src/evaluate.py --config config.yaml --model ihanif/exp_009_small_cv24_vanilla --split test --tag author_checkpoint
#    Expected: WER ~= 24.89, CER ~= 8.08

# 7. Train  (~<T3> on a 16 GB T4)
python src/train.py --config config.yaml --tag exp009_repro_tierB
#    On disconnect: add --resume-from-checkpoint auto

# 8. Evaluate our fine-tuned model (~<T4>)
python src/evaluate.py --config config.yaml --model results/exp009_repro_tierB/best --split test --tag finetuned_small

# 9. Figures and tables
python src/visualize.py --config config.yaml
```

**Expected outputs and tolerances**

| Artefact | Our value | Acceptable reproduction |
|---|---|---|
| Zero-shot WER | `<E1>`% | ± 1 pp (greedy decoding is near-deterministic, but degenerate loops on zero-shot output are sensitive) |
| Author-checkpoint WER | `<E0>`% | ± 0.2 pp |
| Our fine-tuned WER | `<E2>`% | ± 1.5 pp on identical hardware; up to ± 3 pp on a different GPU model |
| CER | `<E2 CER>`% | ± 0.5 pp |
| Best checkpoint step | `<step>` | ± 500 steps (early stopping has a plateau) |
| Total wall clock | `<T>` h | hardware-dependent |

**Hardware floor:** 16 GB VRAM for `whisper-small` full fine-tuning at per-device batch 8 with gradient checkpointing and fp16. With less, switch to `whisper-base` (`model.name: openai/whisper-base` in `config.yaml`; the paper's comparable target is then 27.13% WER from exp_008) or to the LoRA configuration.

**Known deviations from the original paper** are enumerated in Table `<N>` of Section 8 with a reason and a reproducibility impact for each; the five that matter are training-data volume (~12 h vs 113.4 h), learning rate (1e-5 vs 1e-4), precision (fp16 vs fp32), effective batch (32 vs 64), and the re-implemented training script.
