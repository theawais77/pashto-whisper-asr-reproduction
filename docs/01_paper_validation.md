# Step 1 — Base Paper Selection and Validation

**Verdict: ELIGIBLE as base paper, with three documented risks (dataset access, no public training repo, non-peer-reviewed preprint).**

Every field below was verified against the arXiv landing page, the arXiv HTML full text, and the author's Hugging Face repositories on 2026-10-27. Nothing here is inferred or filled in from memory.

## 1.1 Verified paper record

| Field | Verified value |
|---|---|
| Exact title | Fine-tuning Whisper for Pashto ASR: strategies and scale |
| Authors | Hanif Rahman (single author) |
| Submission date | 7 April 2026 (v1) |
| Venue | arXiv preprint, cs.CL. **No journal or conference venue is listed.** |
| arXiv ID | 2604.06507 |
| DOI | 10.48550/arXiv.2604.06507 |
| URL | https://arxiv.org/abs/2604.06507 |
| Full text | https://arxiv.org/html/2604.06507v1 |
| Age | ~6 months old → **satisfies "published within the last 1–2 years"** |
| Task type | **Prediction** (sequence prediction / seq2seq transcription), not classification |
| Model family | Whisper encoder–decoder Transformer: `whisper-base` (74.4M), `whisper-small` (244M), `whisper-large-v3-turbo` (809M) |
| Main metric | Word Error Rate (normalised WER); secondary: orthographic WER, CER |

### Research objective (as stated by the paper)

Pashto is absent from Whisper's pre-training mixture, so off-the-shelf Whisper emits Arabic / Dari / Urdu script and exceeds 100% WER. The paper therefore asks two questions:

1. **Strategy** — which fine-tuning strategy adapts Whisper to Pashto best: full (vanilla) fine-tuning, LoRA, a partially frozen encoder, or multistage Urdu→Pashto transfer?
2. **Scale** — at ~113 h of data, how much does model size buy, i.e. is `whisper-small` the practical optimum against `whisper-base` and `whisper-large-v3-turbo`?

Headline conclusions: **full fine-tuning wins by a large margin** (21.22% WER on CV20 with `whisper-base`), and **`whisper-small` is the practical optimum** on CV24, because `large-v3-turbo` buys only 1.52 pp more WER for 3.3× the parameters.

## 1.2 Reported results we must reproduce against

**Table 1 — Strategy comparison, `whisper-base`, Common Voice Pashto v20 test (N = 2,800)**

| Strategy | Exp | Trainable | WER norm % | WER ortho % | CER % | Δ vs vanilla (pp) |
|---|---|---|---:|---:|---:|---:|
| Vanilla full FT | exp_001 | 74.4M (100%) | **21.22** | 22.56 | 8.52 | — |
| Frozen 2/6 encoder layers | exp_003 | ~56M (75%) | 35.98 | 38.81 | 13.30 | +14.76 |
| LoRA r=64 | exp_002 | 2.36M (3.15%) | 54.58 | 57.77 | 17.21 | +33.36 |
| Multistage Urdu→Pashto | exp_004 | 74.4M (100%) | 65.78 | 68.67 | 21.63 | +44.56 |

**Table 2 — Model scale, vanilla full FT, Common Voice Pashto v24 test (N = 13,791, ~20.7 h)**

| Model | Exp | Params | Zero-shot WER % | WER norm % | WER ortho % | CER % |
|---|---|---:|---:|---:|---:|---:|
| whisper-base | exp_008 | 74.4M | 171.4 | 27.13 | 30.12 | 8.79 |
| **whisper-small** | **exp_009** | **244M** | **120.4** | **24.89** | **27.73** | **8.08** |
| whisper-large-v3-turbo | exp_010 | 809M | 110.2 | 23.37 | 26.77 | 7.64 |

**Augmentation ablation (`whisper-base`, CV24):** the matched comparison gives a **7.25 pp** WER benefit from speed perturbation + Gaussian noise (27.13 with augmentation vs 34.38 without, both at LR 5e-5 / effective batch 32 / BF16). The unmatched comparison gives only 1.50 pp — the paper flags this itself.

**Error analysis (100-utterance samples):** substitutions are ~70–78% of all word errors. Dominant failure modes are word-final suffix confusion (masculine `-ay` vs feminine `-a`) and retroflex substitutions involving /ts/.

## 1.3 THE exact experiment we reproduce

> **exp_009 — full (vanilla) fine-tuning of `openai/whisper-small` on Common Voice Pashto v24 (train + dev combined), evaluated on the held-out CV24 test split.**
>
> **Reproduction targets: WER_norm = 24.89%, CER = 8.075%, best checkpoint = step 8500 (epoch 5.85).**
>
> **Paired baseline to reproduce: zero-shot `whisper-small` on the same test set = 120.4% WER.**

This is the right target because:

- it is the paper's own recommended operating point ("whisper-small is the practical optimum");
- `whisper-small` is already the model our topic selected, with `whisper-base` as the stated fallback — and `whisper-base` maps onto exp_008, so the fallback is also covered by a published number (27.13% WER);
- the published checkpoint `ihanif/exp_009_small_cv24_vanilla` lets us cross-check our evaluation harness against the author's own weights. That gives us a **second, independent reproduction axis** no other experiment offers: if our harness scores his checkpoint at 24.89%, our metric code is validated, and any remaining gap is attributable to training, not measurement.

**Published hyperparameters for exp_009 (everything the paper actually states):** base `openai/whisper-small`; LR 1e-4; effective batch 64; FP32; max 15 epochs; gradient checkpointing enabled; early-stopping patience 3 evaluations; best checkpoint by minimum normalised validation WER; seed 42; GPU NVIDIA RTX 4000 Ada (20 GB), RunPod; Python 3.10, PyTorch 2.x / CUDA 12, Transformers ≥ 4.40, Datasets ≥ 2.18; `Seq2SeqTrainer` wrapped by `fine_tune_whisper_enhanced.py`; TensorBoard logging; metrics via `evaluate` with the HF `BasicTextNormalizer`; `jiwer` v3 for error analysis.

## 1.4 Dataset availability — verified, and this is risk #1

| Resource | Status on 2026-10-27 |
|---|---|
| `mozilla-foundation/common_voice_24_0` on Hugging Face | **Empty.** The card states that effective October 2025, Common Voice datasets are available *exclusively* through Mozilla Data Collective. The v16/v17 mirrors are in the same state. |
| `ihanif/common-voice-pashto-24` (the paper's CV24 mirror) | **HTTP 401.** Does not appear in the author's public dataset list → private or gated. |
| `ihanif/pashto_augmented_speech` (the paper's CV20 corpus) | **HTTP 401.** Same situation. |
| Official Common Voice Pashto v24 | **Available** from Mozilla Data Collective: https://datacollective.mozillafoundation.org/datasets/cmj8u3pnb00llnxxbfvxo3b14 (free account required). Mozilla's own v24 announcement reports ~3,000 h total and ~1,000 h validated for Pashto, with ~5,200 additional speakers contributed in about three months. |
| Third-party Kaggle mirror of CV 24.0 Pashto | **Available**, CC0-1.0: 2,768.7 h total, 975.89 h validated, 2,407,799 clips, 6,654 speakers. A re-upload, **not** an official Mozilla listing — usable as a convenience mirror on Kaggle, but cite the official source as primary. |

**Consequence.** We cannot load the author's exact dataset objects. We rebuild them from the official Common Voice v24 Pashto release using the official `train.tsv` / `dev.tsv` / `test.tsv` split files — which is exactly how the author derived his mirror in the first place. This preserves the official split procedure and is a documented, defensible deviation, not a shortcut. Record the downloaded archive's SHA-256 and the per-split row counts in the report.

## 1.5 Code and checkpoint availability — risk #2

**Checkpoints: public and usable.** Eleven model repos under `ihanif`, Apache-2.0:

`exp_001_base_vanilla`, `exp_001_base_vanilla-gguf`, `exp_002_base_lora`, `exp_002_base_lora_incomplete`, `exp_003_base_frozen`, `exp_004_base_multistage_urdu`, `exp_008_base_cv24_vanilla`, `exp_008b_base_cv24_vanilla_no_aug`, `exp_008b_base_cv24_vanilla_no_aug_low_wer`, **`exp_009_small_cv24_vanilla`**, `exp_010_turbo_cv24_vanilla`.

The `ihanif/exp_009_small_cv24_vanilla` card independently confirms the paper: base `openai/whisper-small`, trained on CV24 train + dev (~113 h), evaluated on the 13,791-sample test split, **WER 24.89 / CER 8.075, best checkpoint-8500 at epoch 5.85**, Apache-2.0, ~0.2B params, F32 safetensors.

**Training and evaluation code: NOT publicly linked.** The paper names `fine_tune_whisper_enhanced.py` and an `experiments/scripts/` directory but gives no repository URL, and the abstract's claim that evaluation scripts are released on Hugging Face did not resolve to any reachable script repository. We therefore **re-implement** the method from the paper's described configuration using the standard `Seq2SeqTrainer` recipe, and we log that re-implementation as the single largest methodological deviation.

A separate public repository, `HarisMajeed05/pashto-asr` (https://github.com/HarisMajeed05/pashto-asr), implements the same Whisper-on-Common-Voice-Pashto pipeline (`scripts/prepare_data.py`, `scripts/fine_tune.py`, `scripts/run_eval.py`, `scripts/common.py`) and is a reasonable structural reference. It is **unrelated to the paper and reports no WER/CER numbers** — cite it as a third-party implementation reference only, never as the paper's official code.

## 1.6 Missing information and reproduction risks — declare every one of these in the report

| # | Issue | Impact | Our handling |
|---|---|---|---|
| 1 | Both of the paper's dataset mirrors return 401 | Cannot byte-match the training corpus | Rebuild from the official CV24 Pashto TSV splits; publish checksums and row counts |
| 2 | No public training/eval script | Method must be re-implemented | Re-implement with `Seq2SeqTrainer`; publish our scripts |
| 3 | **Optimizer is never stated** | LR 1e-4 means very different things under different optimizers | Assume `adamw_torch` (Transformers default); record as an explicit assumption |
| 4 | Warmup unstated for exp_009 (only exp_010 gives warmup ratio 0.1) | Early-training trajectory differs | Use 500 warmup steps; record as a change |
| 5 | The text references a "Table 4" configuration table that does not appear in the paper | The full config is published nowhere | Reconstruct from prose; mark every reconstructed field as such |
| 6 | CV24 dev and test are both reported as 13,791 rows | Possible duplicated or leaked split | Verify row counts ourselves from the official TSVs; report exactly what we find, including any overlap |
| 7 | Zero-shot was evaluated on N = 13,643 but the test split is N = 13,791 | Baseline and fine-tuned numbers are not on an identical set | Evaluate zero-shot on the *identical* set as fine-tuned; note the paper's discrepancy |
| 8 | `whisper-small` listed as 244M in Table 2 but 307M in Appendix C.3 | Minor internal inconsistency; 244M is the correct figure for whisper-small | Report the true parameter count from our loaded model |
| 9 | No Pashto-specific text normalisation described for the CV24 arm | WER depends entirely on `BasicTextNormalizer`, which strips punctuation but performs **no** Arabic-script canonicalisation | Mirror the paper exactly (normalise only at metric time); add our own Pashto normalisation as a separate **ablation**, never as the main run |
| 10 | Single-author arXiv preprint, not peer reviewed | Lower evidential weight | State plainly in Limitations |
| 11 | The paper's own stated limitation: CV20 and CV24 arms are not comparable, and exp_008/009/010 differ in augmentation, LR *and* batch size | Table 2 is **not** a controlled scale ablation and must not be presented as a scaling law | Reproduce exp_009 only; quote the paper's own caveat verbatim in Discussion |
| 12 | **Compute.** exp_009 used 113 h of audio over ~8,500 steps at effective batch 64 on a 20 GB RTX 4000 Ada | Not reachable on free Colab/Kaggle inside session limits | Run a **scaled-down reproduction** on a seeded subset and report the resulting WER gap honestly as a compute-induced deviation (see `docs/04` and `docs/05`) |
| 13 | LoRA tested at a single configuration (r=64, Q+V, α=128) — the paper says so itself | "LoRA is bad for Pashto" does **not** generalise | If we run LoRA, present it as a single-point confirmation only |

## 1.7 Companion and fallback papers (all verified)

- **Companion zero-shot benchmark, same author** — "Benchmarking Multilingual Speech Models on Pashto: Zero-Shot ASR, Script Failure, and Cross-Domain Evaluation", Hanif Rahman, arXiv:2604.04598, 6 April 2026, DOI 10.48550/arXiv.2604.04598. Reports zero-shot Whisper WER from ~90% to 297% across seven model sizes on FLEURS Pashto and a filtered CV24 subset, with `whisper-medium` reaching 461% WER on CV24 due to decoder looping. **This is the correct citation for our zero-shot baseline framing.**
- **Corpus paper** — "Pashto Common Voice: Building the First Open Speech Corpus for a 60-Million-Speaker Low-Resource Language", arXiv:2603.27021.
- **Cross-language analogue with published corpus + fine-tune** — "BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Language", arXiv:2606.03504.

Together these form a coherent citation cluster: corpus → zero-shot failure → fine-tuning fix. Our reproduction sits on the third.

**If the instructor insists on a peer-reviewed venue**, arXiv:2604.06507 will not satisfy that requirement, because no venue is listed. In that case the fallback is BaltiVoice (arXiv:2606.03504) — same task family, same metric, published corpus — but it is also a preprint, so confirm the requirement before switching. Switching the base paper changes nothing in Steps 2–9 except the target numbers.

## 1.8 Administrative discrepancy to resolve with the instructor

The assignment header says **"Work in a group of exactly three students"**, while the final checklist in the same brief says **"Group has exactly two students"**. These contradict each other. Confirm the group size on GCR before submitting, and make the title page and the contribution statement match whatever the instructor confirms. Do not guess.
