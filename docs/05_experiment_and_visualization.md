# Steps 5 & 6 — Reproduction Experiment and Visualizations

## 5.1 The experiment, defined precisely

Four runs. The first three are mandatory; E0 is cheap and worth more than any of them.

| ID | Run | Model | Train data | Test data | Purpose |
|---|---|---|---|---|---|
| **E0** | Harness validation | `ihanif/exp_009_small_cv24_vanilla` (author's weights) | — | CV24 test | Prove our metric code reproduces 24.89% / 8.08%. Isolates measurement error from training error |
| **E1** | Zero-shot baseline | `openai/whisper-small`, untouched, `language="ps"` | — | CV24 test | Reproduce the paper's 120.4% WER and the script-failure phenomenon |
| **E2** | Fine-tuned reproduction | `whisper-small` full FT | CV24 train subset (Tier B ~12 h) | CV24 test | **The reproduction.** Compare against 24.89% |
| **E3** | Ablations (as time allows) | | | | |
| E3a | LR sensitivity | full FT @ LR 1e-4 (paper's value) | same as E2 | CV24 test | Measures the cost of our LR change instead of assuming it |
| E3b | Pashto normalisation | re-score E2 predictions with the canonicalisation map | — | CV24 test | Quantifies how much of the paper's WER is orthographic variation |
| E3c | Augmentation | full FT + speed/noise, p=0.7 | same as E2 | CV24 test | Tests the paper's +7.25 pp claim on `small` (paper measured it on `base`) |
| E3d | LoRA | r=64, α=128, Q+V, dropout 0.1 | same as E2 | CV24 test | Single-point confirmation of the paper's LoRA finding |
| E3e | Data scaling | Tier A / B / C | — | CV24 test | WER-vs-hours curve → explains the gap quantitatively |

**Run order:** E1 → E0 → smoke-test E2 (Tier A) → E2 (Tier B) → E3. Run E1 first because it needs no training and immediately confirms the dataset is loading correctly; if zero-shot WER is not above 100%, something is wrong with the data, not the model.

### E1 — zero-shot protocol
Load `openai/whisper-small` unmodified. Set `language="ps"`, `task="transcribe"`, greedy decoding, `max_new_tokens=225`. Decode all 13,791 test clips. Expected: WER ≫ 100% with heavy insertion counts and output in the wrong script.

**Report the script-failure statistic, not just WER.** The companion paper's central finding is that no Whisper model emits Pashto-script output in more than 0.8% of utterances. Measure it: the fraction of hypotheses whose characters fall in the Pashto-distinctive set `{ټ ډ ړ ڼ ږ ښ ځ څ ګ ۍ}`. A one-line statistic that reproduces a headline claim from a companion paper is worth more to the grader than another decimal place of WER.

### E2 — fine-tuned protocol
Per `docs/04`, Tier B. After training, evaluate the best checkpoint on the identical 13,791-clip test set with identical decoding settings to E1 — same beams, same `max_new_tokens`. Any decoding mismatch between E1 and E2 invalidates the comparison.

Run the final evaluation **3×** and report mean ± std.

## 5.2 Results table (the required format, pre-filled with verified paper values)

Every `TBD` is a cell to fill from an actual run. **Do not pre-fill any of them.** Fabricated numbers are the one thing that fails this assignment outright.

| Experiment | Dataset/Split | WER | CER | Original Paper WER | Difference | Notes |
|---|---|---:|---:|---:|---:|---|
| E1 Zero-shot whisper-small | CV24 Pashto test (N=13,791) | TBD | TBD | 120.4 | TBD | Paper evaluated zero-shot on N=13,643, not 13,791 — noted as risk #7 |
| E0 Author checkpoint `exp_009` | CV24 Pashto test (N=13,791) | TBD | TBD | 24.89 | TBD | Harness validation. Difference ≈ 0 ⇒ our metric code is correct |
| **E2 Our fine-tuned whisper-small** | **CV24 Pashto test (N=13,791)** | **TBD** | **TBD** | **24.89** | **TBD** | **Main reproduction.** Tier B: ~12 h vs paper's 113.4 h; LR 1e-5 vs 1e-4; fp16 vs fp32; eff. batch 32 vs 64 |
| E3a LR 1e-4 (paper LR) | CV24 Pashto test | TBD | TBD | 24.89 | TBD | Isolates the LR deviation |
| E3b + Pashto normalisation | CV24 Pashto test | TBD | TBD | 24.89 | TBD | Not paper-comparable; measures orthographic-variation share |
| E3c + augmentation | CV24 Pashto test | TBD | TBD | 24.89 | TBD | Paper's +7.25 pp claim was measured on `whisper-base`, not `small` |
| E3d LoRA r=64 | CV24 Pashto test | TBD | TBD | 54.58 (base/CV20) | TBD | Cross-arm, cross-model — qualitative only |
| E3e Tier A (0.5 h) | CV24 Pashto test | TBD | TBD | 24.89 | TBD | Scaling point |
| E3e Tier C (25 h) | CV24 Pashto test | TBD | TBD | 24.89 | TBD | Scaling point |
| Reference: paper exp_008 `base` | CV24 Pashto test | — | — | 27.13 | — | Fallback-model target if we run `whisper-base` |
| Reference: paper exp_010 `turbo` | CV24 Pashto test | — | — | 23.37 | — | Paper's upper bound |

**Difference convention — state it in the caption:**

```
Difference (pp) = WER_ours - WER_paper      (positive = our model is worse)
Relative difference (%) = 100 * (WER_ours - WER_paper) / WER_paper
```

Report both. A 12 pp absolute gap is also a ~48% relative gap, and the report should not let the reader pick whichever framing they like.

Also record, for every row: effective batch, precision, LR, decoding (greedy/beam), training hours, total optimizer steps, best step/epoch, wall clock, GPU. Three identical-looking WER rows that differ only in beam width is a classic marks-losing ambiguity.

## 5.3 Hard separation of paper results vs ours

Mechanical rules, applied without exception:

1. The column header is literally **"Original Paper WER"** and is populated **only** from arXiv:2604.06507 Tables 1, 2 and 5.
2. In the report body, every paper number is written as `24.89% [1]` with the citation attached.
3. Our numbers are written as "our reproduction obtained X%" with a pointer to the run tag.
4. `results/metrics.csv` has a `source` column with exactly two values: `paper` or `ours`. The figure scripts colour by that column, so the distinction is visible in the plots, not just in the prose.
5. No chart may interpolate, average or combine a paper value with one of ours into a single mark.

## Step 6 — Visualizations

Nine figures. All generated by `src/visualize.py` from CSV/JSON on disk, never hand-drawn, so they regenerate from the committed artefacts.

| # | Figure | File | Source data | What it must show |
|---|---|---|---|---|
| 1 | Training loss curve | `results/training_loss.png` | `training_log.csv` | train loss vs step. Mark the best checkpoint with a vertical line |
| 2 | Validation loss curve | `results/validation_loss.png` | `training_log.csv` | eval loss vs step, plus eval WER on a twin right-hand axis. **This is the figure that earns discussion** — the paper's Table 2 shows loss and WER are non-monotonic across runs, so if our curves show the same decoupling, we have reproduced a structural property of the paper, not just a number |
| 3 | WER: zero-shot vs fine-tuned | `results/wer_zeroshot_vs_finetuned.png` | `metrics.csv` | Two bars (E1, E2) with a horizontal line at 100% marking "worse than emitting nothing". Annotate the absolute drop |
| 4 | **WER: ours vs paper** | `results/wer_comparison.png` | `metrics.csv` | **The required comparison figure.** Grouped bars: paper 24.89 vs ours, and paper 120.4 vs our zero-shot. Different hatching/colour per `source`. Difference in pp annotated above the pair. A horizontal dashed reference line at 24.89 |
| 5 | CER comparison | `results/cer_comparison.png` | `metrics.csv` | Same structure for CER (paper 8.075). Add the WER/CER ratio as a text annotation — paper ratio ≈ 3.1 |
| 6 | Error-type breakdown | `results/error_breakdown.png` | `predictions_*.csv` | Stacked bars of substitutions / deletions / insertions as % of total word errors, for E1 and E2 side by side, against the paper's reported ~70–78% substitution share. Insertions should dominate for zero-shot (that is what pushes WER past 100%) and substitutions for fine-tuned |
| 7 | Sample predictions | `results/sample_predictions.csv` + a report table | `predictions_*.csv` | 10–20 seeded utterances: reference, zero-shot hypothesis, fine-tuned hypothesis, per-utterance WER. Sample from the best/median/worst terciles so it is demonstrably not cherry-picked. **Ensure the PDF embeds a font with Pashto coverage (Noto Naskh Arabic / Noto Nastaliq Urdu) and renders RTL correctly** — otherwise this table prints as boxes and the whole qualitative section is lost |
| 8 | WER vs training hours | `results/wer_vs_hours.png` | `metrics.csv` (E3e) | Log-x scatter of Tier A/B/C with the paper's 113.4 h / 24.89% point marked distinctly. This figure **explains the gap** rather than apologising for it, and is the strongest single argument that the method reproduces even though the number does not match |
| 9 | Paper figure we reproduce | `results/fig_strategy_or_scale.png` | `metrics.csv` | Reproduce the paper's model-scale comparison (base 27.13 / small 24.89 / turbo 23.37) as published bars, overlaying our measured small point. If `whisper-base` is also run, we have two of the three points measured ourselves |

### Figure conventions
- Identical y-axis limits across figures 3/4 so bar heights are visually comparable.
- Paper bars: hatched, muted. Our bars: solid, saturated. One legend entry each, labelled "Rahman (2026), reported" and "This reproduction".
- Every axis labelled with units; "WER (%)" not "WER".
- 300 dpi PNG, and also emit PDF for vector quality in the report.
- No y-axis truncation on the WER comparison — a y-axis starting at 20 would visually exaggerate a 2 pp gap. Start at 0.
- Caption every figure with the run tag and the date so figures are traceable to runs.

## 5.4 Interpreting the gap (prepare this argument before you see the number)

Ranked by expected contribution to the gap:

1. **Training data volume** — ~12 h vs 113.4 h. Dominant. Figure 8 quantifies it.
2. **Learning rate** — 1e-5 vs 1e-4. E3a measures it directly.
3. **Effective batch / precision** — 32 fp16 vs 64 fp32. Smaller, noisier updates.
4. **Total optimizer steps** — ~3–4k vs 8,500.
5. **Dev held out** — ~15% fewer training rows than the paper, by design.
6. **Corpus rebuild** — we rebuilt from official TSVs; the author's mirror may contain additional filtering we cannot see (401).
7. **Unknowns** — optimizer, warmup, beam width, max generation length, the missing "Table 4".
8. **Hardware non-determinism** — ±0.1–0.3 pp. Negligible, but report the 3-run std so the reader can see it is negligible.

**The reproducibility verdict depends on E0, not E2.** If E0 reproduces 24.89% on the author's checkpoint, then our measurement pipeline is correct and the paper's *reported result* is verified. E2 then tests whether the *training recipe* is reproducible at our compute budget. Separating those two claims is the single most important analytical move in the report — make it explicitly, in those terms.
