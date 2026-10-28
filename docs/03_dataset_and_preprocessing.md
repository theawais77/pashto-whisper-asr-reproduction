# Step 3 — Dataset and Preprocessing Plan

## 3.1 Dataset identity

| Field | Value |
|---|---|
| Name | Mozilla Common Voice — Scripted Speech, Pashto (locale `ps`) |
| Version | **24.0** (matches the paper's CV24 arm) |
| License | **CC0-1.0** (public domain dedication) |
| Official download | https://datacollective.mozillafoundation.org/datasets/cmj8u3pnb00llnxxbfvxo3b14 |
| Release announcement | https://discourse.mozilla.org/t/release-live-mcv-scripted-speech-24-0-and-spontaneous-speech-2-0/146740 |
| Project page | https://commonvoice.mozilla.org/ |
| Scale (Mozilla, v24, Pashto) | ~3,000 h total, ~1,000 h validated |
| Scale (Kaggle mirror metadata) | 2,768.7 h total, 975.89 h validated, 2,407,799 clips, 6,654 speakers |
| Access requirement | Free Mozilla Data Collective account + accepting the dataset terms |
| Convenience mirror (Kaggle) | https://www.kaggle.com/datasets/ataullahaali/common-voice-scripted-speech-24-0-pashto — CC0-1.0, unofficial re-upload |

**Critical access note.** As of October 2025 the `mozilla-foundation/common_voice_*` Hugging Face repos are **empty**; Common Voice is distributed exclusively through Mozilla Data Collective. Any tutorial that does `load_dataset("mozilla-foundation/common_voice_24_0", "ps")` will fail. The paper's own mirrors (`ihanif/common-voice-pashto-24`, `ihanif/pashto_augmented_speech`) return **HTTP 401**. Plan around this; do not discover it the night before the deadline.

### Recommended acquisition route

1. **Primary** — register on Mozilla Data Collective, accept the terms, download the Pashto `ps` v24 archive. This is the citable, official source.
2. **Kaggle route** — if you run on Kaggle, attach the CC0 mirror to the notebook. Zero download time, no quota burn. Cite it explicitly as an unofficial mirror of the official v24 release, and state in the report that the archive checksum was recorded.
3. **Colab route** — upload the archive to Google Drive once, mount it. Never re-download per session.

Record in the report: archive filename, file size, SHA-256, download date, and per-TSV row counts. That turns an access risk into documented provenance.

## 3.2 Archive layout (standard Common Voice format)

```
cv-corpus-24.0-<date>/ps/
├── clips/                 # MP3, 32 kHz or 48 kHz, variable bitrate
├── validated.tsv          # all clips with >= 2 net up-votes
├── train.tsv              # official split
├── dev.tsv                # official split
├── test.tsv               # official split
├── other.tsv              # unvalidated
├── invalidated.tsv        # net down-voted
├── reported.tsv
└── clip_durations.tsv
```

Columns used: `path`, `sentence`, `client_id`, `up_votes`, `down_votes`, `locale`.

## 3.3 Split policy — follow the official procedure, do not invent one

**Use `train.tsv` / `dev.tsv` / `test.tsv` exactly as shipped.** Common Voice generates these with a speaker-disjoint algorithm: a given `client_id` appears in only one split, and dev/test are built to comparable size (which explains the paper's dev = test = 13,791 and means that figure is plausible rather than an error — but verify it yourself and report what you find).

Never re-split with a random `train_test_split`. That leaks speakers across the boundary, inflates the score, and breaks comparability with the paper's numbers — which would invalidate the entire Step 5 difference calculation.

**The paper's split usage (replicate this structure):**

| Split | Paper rows | Paper use |
|---|---:|---|
| train | 79,089 | training |
| dev | 13,791 | **merged into training** (train + dev = 92,880 utterances, ~113.4 h) |
| test | 13,791 (~20.7 h) | held-out evaluation |

The paper trains on train + dev, which leaves it **no clean validation set** even though it selects the best checkpoint by "normalised validation WER" — an inconsistency it never resolves. Do not copy the ambiguity. Our policy:

- **Train on `train.tsv`** (subset — see 3.4).
- **Validate on a seeded subsample of `dev.tsv`** (1,000 utterances, `seed=42`) for checkpoint selection. Keeping dev out of training is a deliberate, documented deviation that makes our checkpoint selection honest. Note it costs us ~15% of the paper's training data, which works *against* us in the WER comparison — state that, since it strengthens rather than weakens our reproduction claim.
- **Evaluate on `test.tsv`**, untouched, never looked at during development.
- **Verify speaker disjointness yourself**: assert `set(client_id) ∩ set(client_id) == ∅` across all three pairs, and report the result. Also report any exact-duplicate `sentence` overlap between dev and test, given risk #6.

## 3.4 Recommended subset size for Colab / Kaggle

Full exp_009 is **out of reach** on free hardware: 113 h of audio, ~8,500 optimizer steps at effective batch 64, on a 20 GB RTX 4000 Ada. A free Colab T4 is roughly 4–6× slower and the session caps at ~12 h with idle disconnects.

| Tier | Train hours | ≈ utterances | Val | Test | Hardware | Wall clock | Expected WER |
|---|---:|---:|---:|---:|---|---|---|
| **A — smoke test** | 0.5 h | 400 | 200 | 500 | any T4 | ~20 min | ~80–100% (pipeline check only) |
| **B — primary run (recommended)** | 12 h | ~10,000 | 1,000 | full 13,791 | T4 16 GB / P100 | 6–9 h | ~35–45% |
| **C — stretch** | 25 h | ~20,500 | 1,000 | full 13,791 | Kaggle P100/T4 ×2, 2 sessions | 12–18 h | ~30–38% |
| **D — faithful** | 113 h | 92,880 | — | full 13,791 | A100 40 GB (Colab Pro+ / RunPod ~$15) | ~20 h | ~25% (paper: 24.89%) |

**Plan on Tier B, with Tier A first and Tier C only if time allows.** Budget 1–2 h separately for the full-test-set evaluation pass (13,791 clips through whisper-small is the single most expensive non-training step — batch it at 16 and use `fp16`).

**Be explicit about the consequence:** at Tier B we will *not* hit 24.89% WER, and the honest report says so. A ~12–18 pp gap caused by using ~10% of the training data is a **correct, well-explained reproduction outcome**, and it scores better than a fabricated match. If budget allows Tier D even once, run it — a 2–3 pp gap is a much stronger result. Either way, the Tier B → Tier C delta gives a data-scaling trend line, which is itself a reportable figure.

**Mandatory extra run, costs almost nothing:** evaluate the author's published checkpoint `ihanif/exp_009_small_cv24_vanilla` on our test split. If we score 24.89%, our metric harness is validated and every remaining gap is attributable to training compute, not measurement error. If we score something different, that is a finding about the paper. This is the single highest-value hour in the whole project.

## 3.5 Preprocessing pipeline

Mirror the paper where it is specified, and document every place it is not.

### Stage 1 — Row filtering (text side)

| # | Step | Rule | Rationale |
|---|---|---|---|
| 1 | Validated only | Use `train/dev/test.tsv` (already derived from `validated.tsv`) | Paper used official splits; unverified clips are noise |
| 2 | Drop empty transcripts | `sentence.strip() == ""` → drop | Empty labels produce degenerate loss |
| 3 | Drop over-long labels | tokenized length > 448 → drop | Whisper decoder hard limit |
| 4 | Drop digit-only / symbol-only rows | no Arabic-script character present → drop | Not transcribable Pashto speech |
| 5 | Deduplicate | identical `(client_id, sentence)` → keep first | Prevents the same utterance being over-weighted |

### Stage 2 — Audio

| # | Step | Rule | Rationale |
|---|---|---|---|
| 6 | Decode + resample | MP3 → 16,000 Hz | Whisper requirement; paper states 16 kHz |
| 7 | Mono | average channels if > 1 | Whisper feature extractor expects 1-D |
| 8 | Duration floor | drop clips < 1.0 s | Paper's CV20 corpus floor was 1.25 s post-augmentation; too-short clips are usually truncated recordings |
| 9 | Duration ceiling | drop clips > 30.0 s | Whisper's fixed window silently truncates past 30 s, creating label/audio mismatch |
| 10 | Drop silent / near-silent | peak amplitude < 1e-4 → drop | Dead-mic recordings |
| 11 | Log-Mel features | `WhisperFeatureExtractor`, 80 bins, pad/trim to 30 s | Standard; do not hand-roll |

CV24 Pashto averages ~4.4 s per clip, so steps 8–9 remove only a small tail — but log the exact counts removed per rule.

### Stage 3 — Text normalisation (the delicate part)

**The paper describes no Pashto-specific normalisation for the CV24 arm and applies HF `BasicTextNormalizer` only at metric time.** So:

- **Main run:** train on **raw** `sentence` strings. Apply `BasicTextNormalizer` only when computing metrics. This is the faithful configuration and the one whose WER is comparable to 24.89%.
- **Ablation run:** apply the Pashto canonicalisation below to both training labels and metric inputs, and report the WER delta separately. This is *our* contribution; do not let it contaminate the main number.

Pashto canonicalisation map for the ablation (all verified Unicode points):

| Issue | Codepoints | Action |
|---|---|---|
| Arabic yeh → Pashto yeh | `ي U+064A` → `ی U+06CC` | map |
| Alef maksura → Pashto yeh | `ى U+0649` → `ی U+06CC` | map |
| Arabic kaf → Pashto kaf | `ك U+0643` → `ک U+06A9` | map |
| Heh variants | `ه U+0647`, `ۀ U+06C0`, `هٔ` | unify to `ه` |
| Zero-width non-joiner | `U+200C` | remove |
| Zero-width joiner / ZWSP | `U+200D`, `U+200B` | remove |
| Tatweel (kashida) | `U+0640` | remove |
| Harakat / diacritics | `U+064B`–`U+0652`, `U+0670` | remove |
| Arabic-Indic digits | `U+0660`–`U+0669` | map to ASCII `0`–`9` |
| Extended Arabic-Indic digits | `U+06F0`–`U+06F9` | map to ASCII `0`–`9` |
| Arabic punctuation | `، U+060C`, `؛ U+061B`, `؟ U+061F`, `۔ U+06D4` | strip (BasicTextNormalizer already strips ASCII punctuation) |
| Unicode form | — | NFC after all substitutions |
| Whitespace | — | collapse runs to a single space, strip ends |

**Characters that must NEVER be altered** — these are the Pashto-distinctive letters and mangling them destroys the task: `ټ ډ ړ ڼ ږ ښ ځ څ ګ ۍ ئ`.

### Stage 4 — Augmentation (the paper's +7.25 pp finding)

exp_009 itself used **no augmentation** (only exp_008 on `whisper-base` did). So:

- **Main exp_009 reproduction: augmentation OFF.** Faithful.
- **Optional secondary run:** augmentation ON, to test the paper's 7.25 pp claim on `whisper-small` rather than `whisper-base`. Online, probability 0.7, via `audiomentations`:
  - speed perturbation ±10–15%
  - Gaussian noise, SNR 15 dB
  (The paper's CV20 corpus additionally used pitch shift, gain normalisation and reverb offline — do not mix that recipe into the CV24 arm.)

Because augmentation was applied to `whisper-base`, confirming it on `whisper-small` is a genuine extension, not a restatement. Label it as such.

## 3.6 Exact documentation format for every preprocessing step

Emit `results/preprocessing_log.csv` programmatically from `src/prepare_data.py`. Never hand-write it.

```csv
step_id,stage,description,rule,param,rows_in,rows_out,rows_removed,pct_removed,hours_in,hours_out,source,deviation_from_paper,timestamp
1,filter,Validated splits only,use official train/dev/test tsv,-,79089,79089,0,0.00,,,paper Sec 2.2,no,2026-10-27T10:00:00Z
2,filter,Drop empty transcripts,sentence.strip()=="",-,79089,79061,28,0.04,,,our addition,"yes - paper silent",2026-10-27T10:00:04Z
...
```

Columns that matter for marks: `rows_removed`, `pct_removed`, `source` (paper section vs our addition), `deviation_from_paper`. Paste this table straight into report Section 7 — it is the literal answer to requirement (d), "Document each step."

Also emit `results/dataset_stats.json`:

```json
{
  "cv_version": "24.0",
  "locale": "ps",
  "archive_sha256": "<fill>",
  "download_date": "<fill>",
  "splits": {
    "train_full": {"rows": 0, "hours": 0.0, "speakers": 0},
    "train_subset": {"rows": 0, "hours": 0.0, "speakers": 0, "seed": 42},
    "dev_subset": {"rows": 1000, "hours": 0.0, "speakers": 0, "seed": 42},
    "test": {"rows": 0, "hours": 0.0, "speakers": 0}
  },
  "speaker_disjoint": {"train_dev": true, "train_test": true, "dev_test": true},
  "dev_test_sentence_overlap": 0,
  "duration_stats_s": {"min": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0},
  "paper_reference": {"train": 79089, "dev": 13791, "test": 13791, "train_plus_dev_hours": 113.4, "test_hours": 20.7}
}
```

## 3.7 Guaranteeing the split follows the official procedure

Put these four assertions in `src/prepare_data.py` so they run every time and fail loudly:

1. Split membership comes **only** from the shipped TSVs — no `train_test_split`, no shuffle-then-slice across split boundaries.
2. Speaker disjointness verified by `client_id` set intersection across all three pairs; assert empty.
3. Subsetting is **within** a split, with a fixed seed, and the subset's speaker set is recorded. Never sample across splits.
4. The test split is used **once**, at the end. No hyperparameter decision may be informed by test WER. Enforce by keeping test loading in `evaluate.py` only, never in `train.py`.

State all four in the report's Reproducibility section — they are the difference between a reproduction and a number that happens to look good.
