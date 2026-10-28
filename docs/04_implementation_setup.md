# Step 4 — Implementation Setup

## 4.1 Software environment (pinned)

The paper gives floors, not pins (Python 3.10, PyTorch 2.x/CUDA 12, Transformers ≥ 4.40, Datasets ≥ 2.18, PEFT unversioned, `evaluate` unversioned, `jiwer` v3). Floors are not reproducible, so we pin inside those floors and say so.

| Component | Our pin | Paper | Why this pin |
|---|---|---|---|
| Python | 3.10.14 | 3.10 | Exact match |
| PyTorch | 2.4.1 + cu121 | 2.x / CUDA 12 | Inside the stated range; Colab-compatible |
| transformers | 4.44.2 | ≥ 4.40 | Inside range; stable `Seq2SeqTrainer` + Whisper |
| datasets | 2.21.0 | ≥ 2.18 | **Must stay on 2.x** — datasets 3.x removed script-based loading and changed the `Audio` feature API; 3.x will break this pipeline |
| evaluate | 0.4.3 | unversioned | WER/CER metric wrappers |
| jiwer | 3.0.4 | v3 | Matches the paper's v3; backs `evaluate` and gives `process_words()` for the error breakdown |
| accelerate | 0.33.0 | unstated | Required by `Seq2SeqTrainer` |
| peft | 0.12.0 | unversioned | Only if the LoRA ablation is run |
| librosa | 0.10.2 | unstated | Resampling / duration probing |
| soundfile | 0.12.1 | unstated | I/O |
| audiomentations | 0.36.0 | used, unversioned | Only for the augmentation ablation |
| tensorboard | 2.17.0 | used | Paper logs to TensorBoard |
| pandas / numpy | 2.2.2 / 1.26.4 | unstated | TSV handling, metrics |
| matplotlib | 3.9.2 | unstated | Figures |
| ffmpeg | system | unstated | MP3 decode |

Capture the real environment at run time — do not transcribe this table into the report as if it were observed:

```bash
python -c "import sys,torch,transformers,datasets,evaluate,jiwer; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(transformers.__version__, datasets.__version__, evaluate.__version__, jiwer.__version__)" | tee results/environment.txt
pip freeze > results/pip_freeze.txt
nvidia-smi | tee results/gpu.txt
```

Those three files are part of the required execution evidence.

## 4.2 GPU requirement

| Model | Strategy | Min VRAM | Fits free Colab T4 (16 GB)? |
|---|---|---:|---|
| whisper-small | full FT, fp16, batch 8 + grad-ckpt | ~13 GB | **Yes** |
| whisper-small | full FT, fp16, batch 16 + grad-ckpt | ~15.5 GB | Borderline — expect OOM |
| whisper-small | full FT, fp32 batch 64 (paper) | ~40 GB+ | No — needs A100 |
| whisper-small | LoRA r=64, fp16, batch 16 | ~8 GB | Yes, comfortably |
| whisper-base | full FT, fp16, batch 16 | ~7 GB | Yes (the stated fallback model) |

Paper used an RTX 4000 Ada (20 GB) at fp32 effective batch 64. We cannot match that memory footprint on free hardware, which forces the precision and batch-composition changes in 4.4.

## 4.3 Reproducibility controls

Seed 42 everywhere (matches the paper). Centralise in `src/utils.py::set_all_seeds`:

```python
random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
os.environ["PYTHONHASHSEED"] = "42"
transformers.set_seed(42)
# determinism (costs ~10-20% speed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Also fix: `dataloader_num_workers=2` with a seeded `generator`, `group_by_length=False` (length grouping makes batch composition data-order-dependent and is a silent source of run-to-run variance).

**State honestly in the report:** full bitwise determinism is not achievable for `whisper-small` fine-tuning across different GPU models, because cuDNN kernel selection and fp16 accumulation order vary by device. Seeds make *our* runs reproducible on *the same* hardware; they do not make another student's T4 produce identical digits. Run the final evaluation 3× and report mean ± std of WER so the reader can see the measurement noise (expect ±0.1–0.3 pp).

## 4.4 Hyperparameters — paper vs ours, with reasons

**Recommendation: full fine-tuning, not LoRA.** This is both faithful and empirically correct: the paper measured LoRA r=64 at **+33.36 pp worse** than full FT. Run LoRA only as a secondary single-point confirmation if time permits, and present it as that.

| Parameter | Paper (exp_009) | Ours (Tier B) | Changed? | Reason | Reproducibility impact |
|---|---|---|---|---|---|
| Base model | `openai/whisper-small` | same | No | — | None |
| Strategy | vanilla full FT | same | No | Paper's winning strategy | None |
| Training data | CV24 train+dev, 92,880 utt / 113.4 h | CV24 train subset, ~10,000 utt / ~12 h | **Yes** | Free Colab/Kaggle cannot complete 113 h | **Largest single source of WER gap.** Expect +10–18 pp |
| Validation data | train+dev merged (no clean val) | seeded 1,000-utt dev subsample, held out | **Yes** | Honest checkpoint selection; paper's setup is self-contradictory | Slightly fewer training rows; makes our selection trustworthy |
| Test data | CV24 test, 13,791 | same, untouched | No | Required for comparability | None |
| Learning rate | 1e-4 | **1e-5** | **Yes** | 1e-4 is ~10× the standard Whisper fine-tune LR. It is defensible at 113 h / batch 64, but on a ~12 h subset at batch 8 it overfits or diverges. 1e-5 is the HF Whisper reference LR | Moderate. **Run 1e-4 as an ablation if time allows** so the deviation is measured, not assumed |
| Precision | FP32 | **fp16** (bf16 if Ampere+) | **Yes** | fp32 whisper-small at any usable batch exceeds 16 GB | Small; fp16 adds slight numerical noise. Note T4 has no bf16 |
| Per-device batch | — | 8 | **Yes** | VRAM ceiling | Combined with grad-accum below |
| Gradient accumulation | — | 4 | **Yes** | 8 × 4 = **effective 32** | Effective batch 32 vs paper's 64 → noisier gradients. Use 8×8=64 on P100/A100 to match exactly |
| Effective batch | 64 | 32 (64 where VRAM allows) | **Yes** | VRAM | Document which was used per run |
| Epochs | max 15 (best at 5.85) | max 10 | **Yes** | Time budget; early stopping governs anyway | Low — early stopping is the real control |
| Max steps | best at 8,500 | ~3,000–4,000 at Tier B | **Yes** | Smaller dataset → fewer steps/epoch | Fewer total updates. Report actual steps |
| Warmup | not stated | 500 steps | **Yes (filling a gap)** | Paper silent for exp_009 (only exp_010 gives ratio 0.1). 500 is the HF Whisper default | Low, but it **is** an unknown — flag it |
| Weight decay | not stated (0.0 for exp_001) | 0.0 | Assumed | Consistency with the one value the paper gives | Low |
| Optimizer | **never stated** | `adamw_torch`, β 0.9/0.999, ε 1e-8 | Assumed | Transformers default | Moderate — LR 1e-4 under a different optimizer is a different operating point. Flag as risk #3 |
| Grad clipping | not stated | 1.0 | Assumed | Transformers default | Low |
| Gradient checkpointing | on | on | No | Required for VRAM anyway | Slower, numerically identical |
| Early stopping | patience 3 evals | patience 3 evals | No | Match | None |
| Eval interval | 500 steps (implied) | 500 steps | No | Match | None |
| Best-model criterion | min normalised val WER | same (`metric_for_best_model="wer"`, `greater_is_better=False`) | No | Match. **Not** val loss — the paper's own tables show loss and WER are non-monotonic | None |
| Augmentation | none for exp_009 | none for main run | No | Faithful | None |
| Generation max length | not stated | 225 | Assumed | HF Whisper default | Low |
| Beams at eval | not stated | 1 (greedy) | Assumed | Paper silent; greedy is the default and the cheap choice | Low–moderate. Beam 5 typically gains 0.5–1.5 pp. **Report which you used**, and if you run beam 5, report both |
| Language / task tokens | `ps` / `transcribe` | same | No | Essential | None |
| `forced_decoder_ids` | — | set `None`, use `language="ps", task="transcribe"` on the processor | n/a | Required on transformers ≥ 4.40 to avoid a deprecated-arg conflict | None |
| Seed | 42 | 42 | No | Match | None |

### If LoRA is run as the secondary ablation

Match the paper exactly so the comparison means something: r = 64, α = 128, dropout 0.1, target modules `q_proj` and `v_proj` in **both** encoder and decoder, ~2.36M trainable params (3.15%). Paper result on `whisper-base`/CV20: 54.58% WER vs 21.22% full FT. On a small subset, LoRA may actually look *relatively better* than it did in the paper, because its regularisation helps at low data. If that happens, say so — it is a real and interesting qualification of the paper's claim, not a failed reproduction.

## 4.5 Commands

All commands read `config.yaml`; CLI flags override it, and every effective config is dumped to the run directory.

```bash
# 0. environment
conda env create -f environment.yml && conda activate pashto-whisper
# or: pip install -r requirements.txt

# 1. prepare data (writes data/processed/ + results/preprocessing_log.csv + results/dataset_stats.json)
python src/prepare_data.py --config config.yaml --cv-root data/raw/cv-corpus-24.0/ps --train-hours 12 --seed 42

# 2. zero-shot baseline (experiment 1)
python src/evaluate.py --config config.yaml --model openai/whisper-small --split test --tag zeroshot_small

# 3. train (experiment 2)
python src/train.py --config config.yaml --tag exp009_repro_tierB

# 4. evaluate the fine-tuned model
python src/evaluate.py --config config.yaml --model results/exp009_repro_tierB/best --split test --tag finetuned_small

# 5. harness validation against the author's published checkpoint (do not skip)
python src/evaluate.py --config config.yaml --model ihanif/exp_009_small_cv24_vanilla --split test --tag author_checkpoint

# 6. figures + tables
python src/visualize.py --config config.yaml

# 7. single-file inference demo for the appendix
python src/infer.py --model results/exp009_repro_tierB/best --audio data/samples/demo.mp3
```

Resume after a Colab disconnect:

```bash
python src/train.py --config config.yaml --tag exp009_repro_tierB --resume-from-checkpoint auto
```

## 4.6 Checkpointing, logging, outputs

**Checkpointing**
- `save_strategy="steps"`, `save_steps=500` (aligned with `eval_steps` — required when `load_best_model_at_end=True`)
- `save_total_limit=2` — Colab disk is ~70 GB and each whisper-small checkpoint is ~1 GB
- `load_best_model_at_end=True`, then save the winner to `results/<tag>/best/`
- Mirror `results/<tag>/best/` to Google Drive immediately; Colab wipes local disk on disconnect
- Record the winning step and epoch in `results/<tag>/best_checkpoint.json` (the paper's exp_009 won at step 8500 / epoch 5.85 — this is a directly comparable number)

**Logging**
- TensorBoard to `results/<tag>/logs/` (matches the paper)
- `logging_steps=25`
- `results/<tag>/trainer_state.json` holds the full `log_history` — this is the source for the loss and WER curves. Also export `results/<tag>/training_log.csv` with `step, epoch, train_loss, eval_loss, eval_wer, eval_cer, learning_rate, grad_norm`
- `results/<tag>/effective_config.yaml` — the resolved config actually used
- `results/<tag>/run_meta.json` — git commit SHA, start/end time, GPU name, wall-clock seconds, peak VRAM

**Per-evaluation artefacts**
- `results/metrics.csv` — one row per experiment, appended (the Step 5 table)
- `results/predictions_<tag>.csv` — `id, audio_path, reference, hypothesis, reference_norm, hypothesis_norm, wer_utt, cer_utt, n_sub, n_del, n_ins`
- `results/sample_predictions.csv` — 20 seeded rows for the appendix (a mix of best, median and worst utterance WER, so it is not cherry-picked)
