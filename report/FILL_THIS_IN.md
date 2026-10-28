# What I need from you to finish the report

The DOCX is at `report/CS444_A2_PashtoWhisper_Report.docx`. Every blank is marked `[[ like this ]]` and **highlighted yellow**, so in Word you can press Ctrl+F, search `[[`, and walk them all. There are **291 placeholders across 110 distinct values** — but they collapse into 5 batches, and 3 of the batches cost nothing but a copy-paste.

---

## BATCH 1 — You can fill these right now, no GPU needed (5 minutes)

| Placeholder | What to put |
|---|---|
| `[[ Member 1/2/3 full name ]]`, `[[ reg no. ]]` | Your three names + registration numbers. Title page **and** the Contributions table. |
| `[[ submission date ]]` | Whatever GCR says. |
| `[[ GitHub repository URL ]]` | Create the repo, push the `pashto-whisper-asr-reproduction/` folder. Appears on the title page, §9.6 and Appendix A. |
| `[[ commit SHA ]]` | `git rev-parse HEAD` **after** the final push. Pin it — this is what makes the report auditable. |
| `[[ date ]]` (×10, in References) | Access date for each URL. Use your actual submission date. |
| Contributions table rows | Reassign to match who actually did what. Do **not** write "all contributed equally". |

---

## BATCH 2 — After you run `prepare_data.py` (~30 min, CPU only, no GPU)

Run this once the Common Voice archive is downloaded:

```bash
python src/prepare_data.py --config config.yaml --cv-root data/raw/cv-corpus-24.0/ps --train-hours 12 --seed 42 --archive <path-to-archive>
```

Then give me — or paste yourself — these files and I'll fill the tables:

| Source file | Fills |
|---|---|
| `results/dataset_stats.json` | §4.1 archive SHA-256 + size + download date · §4.2 test rows/hours/speakers · §4.3 train rows/hours/speakers, dev rows · §4.5 the three `pass/fail` speaker-disjointness checks and the dev/test overlap count |
| `results/preprocessing_log.csv` | **Appendix B** in full (this is literally assignment requirement (d) — don't skip it) |
| Console output of the command | Appendix A.1 `[prepare_data]` log lines |

---

## BATCH 3 — The cheap experiments, no training (~2 hours on free Colab T4)

**Use the notebook: [`notebooks/batch3_reproduction_colab.ipynb`](../notebooks/batch3_reproduction_colab.ipynb).** Upload it to Colab (or Kaggle — Kaggle is easier, see below) and run top to bottom. It is self-contained; it does not need the repo cloned.

It covers Batch 2 and Batch 3 in one pass: environment capture, config recovery, dataset prep, both evaluations, both of your original analyses, the difference arithmetic, five figures, and a results zip.

Equivalent from the command line, if you prefer scripts:

```bash
# B0 — zero-shot baseline. Paper target: 120.4% WER
python src/evaluate.py --config config.yaml --model openai/whisper-small --split test --tag zeroshot_small

# B — the actual reproduction: author's released weights, our harness. Paper target: 24.89% WER
python src/evaluate.py --config config.yaml --model ihanif/exp_009_small_cv24_vanilla --split test --tag author_checkpoint
```

### Picking a data route — do this first, it decides the platform

| Route | How | Verdict |
|---|---|---|
| **A** — `ihanif/common-voice-pashto-24` on HF | Sign in to a free HF account, open the dataset page. If there's an **"Agree and access"** button it's *gated* → accept, make a read token, paste it into the notebook. If you get **404** it's *private* → dead. | **Check this first, it's 2 minutes.** It's the paper's exact dataset object, so it gives perfect comparability. |
| **B** — Kaggle mirror | Kaggle notebook → Add Input → Datasets → `ataullahaali/common-voice-scripted-speech-24-0-pashto`. Mounts read-only, **no download at all**. | **The reliable fallback, and why Kaggle beats Colab here.** CC0-1.0. Cite it as an unofficial mirror. |
| **C** — Mozilla Data Collective | Download the official `ps` v24 archive → Google Drive → mount in Colab. | Most authentic, slowest. The archive is tens of GB. |

Anonymous access to Route A returns HTTP 401 and HF reports it as *"private or gated"* — I can't tell which without your login. Hence: check Route A, fall back to B.

### The notebook also recovers data the paper never published

Cell 3 downloads two files from the author's checkpoint repo that the paper doesn't mention and that need no authentication:

- `experiment_log.txt` (3.9 MB) — the **complete exp_009 training log**, opening with a full `ARGUMENTS SUMMARY`
- `training_args.bin` — the serialised `TrainingArguments`

This turns most of the "assumed" rows in §5.3 into matched published values: **learning rate 1e-4, true batch 64 with accumulation 1, FP32, warmup 500 steps, weight decay 0.0, 15 epochs, eval every 500 steps, early-stopping patience 5, duration window 0.5–30 s.** No `optim` is passed anywhere, so `adamw_torch` is an inference from the config rather than a guess.

It also recovers **19 evaluation points** — the paper's own WER/loss curve across 6.5 epochs — which cell 12 plots as a reproduced figure. Validation loss bottoms at epoch 2.76 and then *rises* while WER keeps improving to its best at epoch 5.85; that decoupling is why the paper selects on WER, and it's good Discussion material.

**And one finding you should put in §9.** The recovered config says `eval_dataset_splits: ['test']`. The evaluation loop that chose the best checkpoint ran on the **test split** — so the reported 24.89% is selected on the same data it's reported on. The paper calls this "validation WER", but no separate validation set existed. State it factually; it means our held-out-dev design is methodologically cleaner than the paper's.

| Fills | Where |
|---|---|
| Zero-shot WER / WER-ortho / CER | §6.2 Table 1, §7 Table 5, §7.1, Abstract, §10 |
| Author-checkpoint WER / CER + bootstrap CI | §6.2 Table 1 (**the row that decides whether the paper reproduces**), §7, §9.1(a) |
| Pashto-script output rate for both | §6.4 Table 3 — this is one of your two original contributions |
| Insertion share for both | §6.4 Table 3 |
| Normalisation-ablation WER (re-run `evaluate.py` with `--rescore-pashto-norm`) | §6.4 Table 4 — your second original contribution |
| Console output | Appendix A.1, Screenshots 3 and 4 |

**If you only have time for one thing, do Batch 3.** Part B alone answers "did the paper reproduce?" with a real measured number.

---

## BATCH 4 — Your own training run (6–9 hours on free Colab T4)

**Use the notebook: [`notebooks/batch4_training_colab.ipynb`](../notebooks/batch4_training_colab.ipynb).** Self-contained, runs on Colab or Kaggle.

Four things it does that a plain script run does not:

- **Checkpoints to Google Drive and resumes.** Colab *will* disconnect. Re-run cells 1–7 then cell 8 and it picks up from the last checkpoint, optimizer state included.
- **A wall-clock guard** (`MAX_HOURS = 7.0`) stops training cleanly so `load_best_model_at_end` still fires. Being killed at 7.5 h with nothing saved is the failure this prevents.
- **Prints a test-split fingerprint** (cell 5). It must match the one Batch 3 printed, or your Part B and Part C numbers are not comparable. Check it *before* you start training, not after.
- **Picks precision from the hardware.** bf16 on Ampere+, fp16 on T4, fp32 with a smaller batch on P100 — where fp16 Whisper training is NaN-prone and has no tensor-core speedup. If you draw a P100, consider restarting the runtime for a T4.

Equivalent from the command line:

```bash
python src/train.py --config config.yaml --tag exp009_repro_tierB
# disconnected? add --resume-from-checkpoint auto
python src/evaluate.py --config config.yaml --model results/exp009_repro_tierB/best --split test --tag finetuned_small
```

### The one judgement call: learning rate

The paper used **1e-4** at a true batch of 64 in FP32. The notebook defaults to **1e-5**, the HuggingFace Whisper reference rate, because 1e-4 tends to overfit or diverge on a tenth of the data at an accumulated batch of 32 in fp16.

That is a real deviation. If you have GPU time after the main run, re-run with `LEARNING_RATE = 1e-4` and `RUN_TAG = "exp009_repro_lr1e4"`. That is ablation **E3a**, and it converts "we changed the learning rate" into "we changed it and here is what it cost".

### Highest-value optional run

`TRAIN_HOURS = 3.0`, `RUN_TAG = "exp009_repro_tierA"` — about 1.5 h. It gives Figure 7 a second point, so the figure becomes a real data-scaling trend line with a log-linear extrapolation to the paper's 113.4 h instead of a single dot. **That figure is what turns your WER gap from an apology into a measurement**, so this is the best 90 minutes of GPU time left anywhere in the project.

| Source | Fills |
|---|---|
| `results/<tag>/training_log.csv` | §6.3 Table 2 (one row per eval checkpoint) · Figures 1 and 2 |
| `results/<tag>/best_checkpoint.json` | §4.3 best step/epoch · §6.3 · §7 Table 5 (paper: step 8500, epoch 5.85) |
| `results/<tag>/run_meta.json` | §5.1 GPU/VRAM · §6.3 wall clock, peak VRAM · §5.3 hardware row |
| `results/environment.txt`, `pip_freeze.txt`, `gpu.txt` | §5.1 all version placeholders |
| Final eval output | §6.3 test WER/CER · §7 · §7.1 · §9.3 · Abstract · §10 |
| **Decisions you make** | §5.3: which LR you actually ran (1e-5 or 1e-4), fp16 vs bf16, per-device batch × grad-accum, max epochs, warmup steps, greedy vs beam |

Also capture while it runs:
```bash
python -c "import sys,torch,transformers,datasets,evaluate,jiwer; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(transformers.__version__, datasets.__version__, evaluate.__version__, jiwer.__version__)" | tee results/environment.txt
pip freeze > results/pip_freeze.txt
nvidia-smi | tee results/gpu.txt
```

---

## BATCH 5 — Figures and screenshots (30 min)

```bash
python src/visualize.py --config config.yaml --tag exp009_repro_tierB
```

Produces 8 PNGs. Paste each into its dashed yellow box in §8. **Figure 7 (WER vs training hours) needs at least two training runs at different `--train-hours`** — if you only do one, delete that figure box and say so in §9.3.

Then take **8 screenshots** from the executed Colab notebook for Appendix A.2. The captions are already written; the boxes tell you exactly what each one must show. Don't crop out the timestamps — they're the evidence.

Also: §8.1 sample predictions table, from `results/sample_predictions.csv`.

---

## Three things that will cost you marks if you skip them

1. **Pashto font.** The sample-predictions table (§8.1) will print as empty boxes unless you set those three columns to **Noto Naskh Arabic** and mark them right-to-left. Export the PDF and open it on a *different* machine before submitting. Friend's report had no RTL text so they never hit this; you will.
2. **Update the TOC.** Click in the Contents table → F9 → "Update entire table". Then delete the red instruction line under it. Delete the two other red instruction lines too (under References, under §8.1).
3. **Verify references [5], [6], [8], [9], [10], [13].** I wrote those page ranges from knowledge, not from the publisher page. [1]–[4] I verified live. Check them on ACL Anthology / PMLR / NeurIPS Proceedings and fix anything wrong.

---

## Fastest viable path if you're short on time

Batch 1 → Batch 2 → **Batch 3** → Batch 5 (figures 3,4,5,6 only) → submit with §6.3 marked "not completed, see Limitations".

That gets you: a verified reproduction of the paper's headline number, a measured zero-shot baseline, both of your original analyses, four figures, and an honest limitation statement. It is a complete report. Batch 4 upgrades it from "we verified the paper" to "we also retrained it", which is better but is not what the grading hinges on.
