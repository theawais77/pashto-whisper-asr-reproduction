# Step 11 — Final Submission Checklist

Tick only what is verifiably true. The last column names the artefact a grader can open to confirm it.

## A. Eligibility and administration

| # | Item | Status | Evidence |
|---|---|---|---|
| A1 | **Group size confirmed with the instructor** — the brief says "exactly three students" in the header and "exactly two students" in the checklist. **These contradict; resolve on GCR before submitting.** | ☐ | GCR post / instructor reply |
| A2 | Title page carries university, department, assignment no., course, semester, section, marks, instructor | ☐ | Report §1 |
| A3 | All member names and registration numbers listed | ☐ | Report §1 |
| A4 | Per-member contribution statement, specific, with named sections and files | ☐ | Report §1 table |
| A5 | Every member has independently re-run `src/evaluate.py` and can explain every number | ☐ | §17; verbal defence |
| A6 | One single PDF, named `CS444_A2_BSCS7B_ShiftII_<Group>_PashtoWhisper.pdf` | ☐ | GCR upload |
| A7 | Pashto script renders correctly in the exported PDF, checked on a second machine | ☐ | Appendix F |

## B. Paper selection (requirement b)

| # | Item | Status | Evidence |
|---|---|---|---|
| B1 | Paper published within the last 2 years — **April 2026, ~6 months old ✔** | ☐ | Report §3 |
| B2 | Task is classification or prediction — **prediction (autoregressive seq2seq) ✔**, justified formally | ☐ | §3, §4 |
| B3 | Exact title stated | ☐ | §3 |
| B4 | Authors stated — Hanif Rahman (single author) | ☐ | §3 |
| B5 | Publication date stated — 7 April 2026 | ☐ | §3 |
| B6 | DOI **and** URL stated — 10.48550/arXiv.2604.06507 / https://arxiv.org/abs/2604.06507 | ☐ | §3, [1] |
| B7 | Research objective stated in the paper's own terms | ☐ | §3 |
| B8 | Venue stated honestly as a non-peer-reviewed arXiv preprint | ☐ | §3, §16 |
| B9 | Deep-learning model identified — Whisper encoder–decoder Transformer, `whisper-small`, 244M | ☐ | §5 |
| B10 | Main metric and the paper's reported value identified — normalised WER, 24.89% (exp_009) | ☐ | §3, §10 |
| B11 | The exact experiment being reproduced is named — exp_009 | ☐ | §9 |

## C. Dataset and preprocessing (requirement d)

| # | Item | Status | Evidence |
|---|---|---|---|
| C1 | Dataset name, version, licence cited — Common Voice Scripted Speech 24.0, `ps`, CC0-1.0 | ☐ | §6, [4] |
| C2 | Official download source cited, with access date | ☐ | [4] |
| C3 | Any mirror used is cited **as a mirror**, not as the official source | ☐ | [4a] |
| C4 | The paper's dataset mirrors returning HTTP 401 is documented as a reproduction risk | ☐ | §6, §16, §17 |
| C5 | Archive SHA-256 and download date recorded | ☐ | `results/dataset_stats.json` |
| C6 | Train/val/test split documented with exact row counts and hours | ☐ | §7, `dataset_stats.json` |
| C7 | Split follows the official shipped TSVs — no custom re-split | ☐ | §7; assertion in `prepare_data.py` |
| C8 | Speaker disjointness verified across all three split pairs and the result reported | ☐ | `dataset_stats.json` |
| C9 | dev/test sentence-overlap check performed (paper reports dev = test = 13,791) | ☐ | `dataset_stats.json` |
| C10 | 16 kHz mono conversion documented | ☐ | preprocessing log step 6–7 |
| C11 | Empty transcripts removed; over-long labels removed; over-30 s and under-1 s clips removed; silent clips removed — each with counts | ☐ | preprocessing log steps 2, 3, 8, 9, 10 |
| C12 | Unverified / invalidated samples excluded (official splits only) | ☐ | preprocessing log step 1 |
| C13 | Pashto text normalisation policy stated: raw labels for training, `BasicTextNormalizer` at metric time — matching the paper | ☐ | §7, §10 |
| C14 | Special characters, ZWNJ, tatweel, diacritics and digit variants handled consistently, documented as an **ablation** | ☐ | §7, E3b |
| C15 | Every preprocessing step logged in the documented CSV format with rows-in/out/removed | ☐ | `results/preprocessing_log.csv` |
| C16 | Test split used exactly once, after all decisions | ☐ | §17; test loading confined to `evaluate.py` |

## D. Implementation (requirement e)

| # | Item | Status | Evidence |
|---|---|---|---|
| D1 | Software environment captured from the live run, not transcribed | ☐ | `results/environment.txt`, `pip_freeze.txt` |
| D2 | Python, PyTorch, Transformers, Datasets, evaluate, jiwer versions recorded | ☐ | `environment.txt` |
| D3 | GPU model and VRAM recorded | ☐ | `results/gpu.txt` |
| D4 | Random seed (42) set across random / numpy / torch / CUDA / `PYTHONHASHSEED`, and determinism flags set | ☐ | `src/utils.py` |
| D5 | Cross-device non-determinism acknowledged honestly | ☐ | §8, §17 |
| D6 | All model settings recorded — LR, batch, grad-accum, epochs, warmup, weight decay, precision, early stopping, best-model criterion | ☐ | §8, `effective_config.yaml` |
| D7 | **Three-column deviation table: paper setting / our setting / reason + reproducibility impact** | ☐ | §8 |
| D8 | Assumptions labelled "assumed" — optimizer, warmup, beam width, max gen length, weight decay | ☐ | §5, §8 |
| D9 | Checkpointing configured and the best step/epoch recorded (paper: step 8500, epoch 5.85) | ☐ | `best_checkpoint.json` |
| D10 | Training logged to TensorBoard **and** exported to CSV | ☐ | `results/<tag>/logs/`, `training_log.csv` |
| D11 | Training, evaluation and inference commands all given verbatim | ☐ | §8, Appendix B |
| D12 | Inference script exists and was run on at least one clip | ☐ | `src/infer.py`, Appendix F |
| D13 | Choice of full fine-tuning over LoRA justified from the paper's own +33.36 pp finding | ☐ | §5, §8 |

## E. Evaluation and results (requirement f)

| # | Item | Status | Evidence |
|---|---|---|---|
| E1 | WER used as the main metric, CER as secondary | ☐ | §10 |
| E2 | Metric formulas given with S/D/I/N defined; WER > 100% explained | ☐ | §10 |
| E3 | It is stated that the comparison uses **normalised** WER on both sides | ☐ | §10, §13 |
| E4 | Results presented in the required table format | ☐ | §11 |
| E5 | Zero-shot baseline measured on the identical test set | ☐ | §11, E1 |
| E6 | Zero-shot script-failure statistic reported (reproduces [2]'s headline claim) | ☐ | §11 |
| E7 | **Author's published checkpoint evaluated with our harness (E0)** — validates measurement independently of training | ☐ | §12 |
| E8 | Evaluation repeated 3× and mean ± std reported | ☐ | §11 |
| E9 | No fabricated values anywhere; no `TBD` left in the submitted PDF | ☐ | whole report |

## F. Comparison, visualization, difference (requirement g)

| # | Item | Status | Evidence |
|---|---|---|---|
| F1 | A visualization directly compares our result with the paper's | ☐ | Fig. 4 |
| F2 | Paper bars and our bars visually distinguished, with a legend naming the source | ☐ | Fig. 4, 5, 9 |
| F3 | Y-axis starts at 0 on the WER comparison (no truncated-axis exaggeration) | ☐ | Fig. 4 |
| F4 | **WER difference computed explicitly**, absolute (pp) and relative (%), with the sign convention stated | ☐ | §13 |
| F5 | CER difference also computed (paper 8.075%) | ☐ | §13 |
| F6 | Training and validation loss curves included | ☐ | Fig. 1, 2 |
| F7 | Zero-shot vs fine-tuned WER chart included | ☐ | Fig. 3 |
| F8 | Error-type breakdown (S/D/I) included and compared to the paper's ~70–78% substitution share | ☐ | Fig. 6 |
| F9 | Sample predictions with references included, sampled from best/median/worst terciles | ☐ | Fig. 7 / Appendix F |
| F10 | At least one figure from the original paper reproduced (model-scale comparison) | ☐ | Fig. 9 |
| F11 | WER-vs-training-hours figure included to quantify the compute-induced gap | ☐ | Fig. 8 |

## G. Discussion, limitations, reproducibility (requirement h)

| # | Item | Status | Evidence |
|---|---|---|---|
| G1 | Differences in results explained with ranked, **measured** causes (E3a, E3e), not assertions | ☐ | §15 |
| G2 | Limitations split into: the paper's, its artefacts', and ours | ☐ | §16 |
| G3 | The paper's own stated limitations quoted (single LoRA config, 50-step grid, no dialect stratification, arms not comparable, exp_008/009/010 not a controlled ablation) | ☐ | §16 |
| G4 | The paper's internal inconsistencies we found are documented (missing Table 4, 244M vs 307M, dev = test = 13,791, zero-shot N = 13,643 vs 13,791, optimizer unstated) | ☐ | §16 |
| G5 | Reproducibility verdict stated, separating "the paper's reported result is verifiable" from "the training recipe is reproducible at our compute budget" | ☐ | §17 |
| G6 | What is included is listed, item by item, with paths | ☐ | §17.2 |
| G7 | What is **not** included is listed with a reason for each | ☐ | §17.3 |
| G8 | Exact step-by-step reproduction instructions with expected values and tolerances | ☐ | §17.4 |
| G9 | Hardware floor stated, with a documented fallback (`whisper-base` → paper target 27.13%) | ☐ | §17.4 |

## H. Code and evidence

| # | Item | Status | Evidence |
|---|---|---|---|
| H1 | Public repository link included | ☐ | Appendix A |
| H2 | Exact commit SHA for the reported numbers included | ☐ | Appendix A |
| H3 | Repository contains `README.md`, `requirements.txt`, `environment.yml`, `config.yaml`, `src/`, `results/`, `report/` | ☐ | repo tree |
| H4 | `src/` contains `prepare_data.py`, `train.py`, `evaluate.py`, `infer.py`, `utils.py` (+ `visualize.py`) | ☐ | repo tree |
| H5 | `results/` contains `metrics.csv`, `training_loss.png`, `validation_loss.png`, `wer_comparison.png`, `sample_predictions.csv` | ☐ | repo tree |
| H6 | Notebook committed **with cell outputs retained** | ☐ | `notebooks/` |
| H7 | **Genuine execution evidence** — timestamped screenshots of GPU allocation, training progress with step/loss/WER, final evaluation output, `nvidia-smi` | ☐ | Appendix D |
| H8 | Console logs included as text, not only as screenshots | ☐ | Appendix D |
| H9 | Checkpoint published or obtainable, with instructions | ☐ | §17.2, Appendix H |

## I. References

| # | Item | Status | Evidence |
|---|---|---|---|
| I1 | IEEE style throughout | ☐ | §19 |
| I2 | Base paper cited with DOI and URL | ☐ | [1] |
| I3 | Dataset cited with licence and access date | ☐ | [4] |
| I4 | Whisper model paper cited | ☐ | [6] |
| I5 | Transformers, Datasets, PyTorch, evaluate, jiwer cited | ☐ | [8]–[12] |
| I6 | PEFT/LoRA cited **only if** the LoRA ablation was run | ☐ | [13], [14] |
| I7 | Author's checkpoint cited as a distinct source | ☐ | [3] |
| I8 | Related Pashto ASR work cited | ☐ | [2], [16] |
| I9 | Third-party implementation reference cited as unaffiliated | ☐ | [19] |
| I10 | **Page numbers and proceedings titles for [5]–[15] and [18] verified against the publisher page** | ☐ | §19 |
| I11 | Every URL carries an access date; nothing cited that was not read | ☐ | §19 |

---

## Three things most likely to go wrong

1. **Dataset access.** The HF Common Voice repos are empty and the paper's mirrors are 401. Register on Mozilla Data Collective and download **in week one**, not the night before.
2. **Pashto rendering in the PDF.** `pdflatex` will not render Arabic script. Use XeLaTeX/LuaLaTeX with Noto Naskh Arabic, and open the exported PDF on a different machine to confirm.
3. **Colab disconnects mid-training.** Checkpoint every 500 steps to Google Drive and always resume with `--resume-from-checkpoint auto`. A lost 6-hour run costs a day.
