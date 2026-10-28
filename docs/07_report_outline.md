# Step 8 — PDF-Ready Report Outline

Target length 18–25 pages. Single PDF, submitted once per group on GCR. Page budgets are deliberate: the marks sit in Sections 7, 8, 11, 12, 13, 15, 16, 17.

---

## 1. Title Page — 1 p

```
AIR UNIVERSITY ISLAMABAD
Department of Computer Science | Faculty of Computing & AI

ASSIGNMENT # 02
Research Paper Reproduction

Fine-Tuning Whisper for Pashto Automatic Speech Recognition
Using Transfer Learning
A Reproduction of Rahman (2026), arXiv:2604.06507

Course:      CS-444 Deep Learning
Semester:    Fall 2026
Section:     BSCS-VII-B (Shift-II)
Total Marks: 50
CLO-2 | GA-3 | C3 (Apply)
Instructor:  Mr. Qadeer Yasin
Issue Date:  19 October 2026
Submission Date: <date>

Group Members
  <Name>  <Reg. No.>
  <Name>  <Reg. No.>
  <Name>  <Reg. No.>     <- confirm group size with the instructor first
```

**Contribution statement** (required; be specific, no "all members contributed equally"):

| Member | Reg. No. | Contribution | Sections / files owned |
|---|---|---|---|
| `<Name>` | `<Reg>` | Paper selection and validation; dataset acquisition from Mozilla Data Collective; preprocessing implementation and audit log | §3, §6, §7; `src/prepare_data.py`, `src/utils.py` |
| `<Name>` | `<Reg>` | Training implementation; hyperparameter configuration; checkpointing, logging, Colab/Kaggle execution | §5, §8, §9; `src/train.py`, `config.yaml` |
| `<Name>` | `<Reg>` | Evaluation harness; metrics; figures; comparison analysis; reproducibility statement | §10–§14, §17; `src/evaluate.py`, `src/visualize.py` |

Add one line: *"All members reviewed the full pipeline, independently re-ran the evaluation script, and can explain every reported number."* — this is a stated requirement.

## 2. Abstract — 150–200 words, ½ p

Must contain, in order: the base paper and its reported WER; what we reproduced; our WER and CER; the difference in pp; the one-sentence reason for the gap; the reproducibility verdict. No numbers may appear here that are not also in Section 11.

## 3. Paper Details and Research Objective — 1 p
Table of verified metadata (title, author, date, venue, arXiv ID, DOI, URL). The paper's objective and two research questions. The paper's four strategies and three model sizes. Our eligibility justification: published April 2026 (within 2 years), prediction task, dataset and checkpoints identifiable. State that it is a non-peer-reviewed arXiv preprint. Source: `docs/01`.

## 4. The Pashto ASR Prediction Problem — 1½ p
Why ASR is sequence prediction and not classification, with the factorised conditional. Input/output specification. Why Pashto is hard: 60M speakers, low-resource, absent from Whisper pre-training, retroflex and affricate phonemes, Arabic-script orthographic variation, agglutinative word-final morphology. Mapping to CLO-2 (prediction) and taxonomy level C3 (Apply). Source: `docs/02`.

## 5. Whisper and Transfer-Learning Methodology — 2 p
Whisper architecture (encoder conv stack → 12+12 Transformer blocks → 51,865-token BPE head), the multitask token interface, log-Mel front end. Transfer learning: what is retained, what is relearned. The paper's four strategies and why full fine-tuning wins, including the paper's 6-encoder-layer argument against freezing. Loss (teacher-forced cross-entropy, `-100` masking) and optimizer (AdamW — **declare this as our assumption**). Include an architecture figure. Source: `docs/02`.

## 6. Dataset Description — 1½ p
Common Voice Scripted Speech 24.0, locale `ps`, CC0-1.0. Official Mozilla figures (~3,000 h total, ~1,000 h validated). Archive layout and TSV schema. **The access problem**: HF `mozilla-foundation` repos empty since Oct 2025; the paper's two mirrors return 401. How we obtained the data instead, with SHA-256. Paper's split sizes (79,089 / 13,791 / 13,791) vs ours. Duration histogram figure. Source: `docs/03`.

## 7. Data Preprocessing and Splitting — 2 p **(marks-heavy: requirement (d))**
The 11-step pipeline as a table. Embed `results/preprocessing_log.csv` verbatim with rows-in / rows-out / rows-removed per rule. Split policy: official TSVs only, speaker-disjointness assertion result, dev-held-out rationale, dev/test sentence-overlap check (risk #6). Text-normalisation policy: raw labels for training, `BasicTextNormalizer` at metric time only — matching the paper — with our Pashto canonicalisation map presented as an ablation. Source: `docs/03`.

## 8. Model and Implementation Setup — 2 p **(marks-heavy: requirement (e))**
Pinned environment table (captured, not transcribed). GPU used. Seed and determinism flags, with the honest statement about cross-device non-determinism. **The three-column deviation table — paper setting / our setting / reason + reproducibility impact — is the single most important table in the report.** All commands. Checkpointing and logging configuration. Source: `docs/04`.

## 9. Experimental Procedure — 1 p
The E0–E3 run matrix, run order, and why E1 runs first. Decoding settings held identical across E1/E2. Wall-clock cost per run. Source: `docs/05`.

## 10. Evaluation Metrics — 1 p **(requirement (f))**
WER and CER formulas with S/D/I/N defined. Why WER can exceed 100%. The three variants (orthographic / normalised / CER) and the explicit statement that the paper's headline metric — and therefore our comparison metric — is **normalised WER**. Why CER matters for agglutinative Pashto. `evaluate` + `jiwer` implementation. The `BasicTextNormalizer` caveat about Arabic-script canonicalisation. Source: `docs/02`.

## 11. Reproduced Results — 1½ p **(requirement (f))**
The full results table from `docs/05` §5.2, filled from actual runs. Mean ± std over 3 evaluation runs. Zero-shot script-failure statistic. Training summary: total steps, best step/epoch, wall clock, peak VRAM. **Only our numbers in this section.**

## 12. Comparison with the Original Paper — 1½ p **(requirement (g))**
Side-by-side table. Figures 4 (WER comparison), 5 (CER), 9 (reproduced paper figure). The E0 result stated first, because it determines which of the two reproducibility claims is being tested. `paper` vs `ours` visually distinguished in every chart.

## 13. Difference in the Main Evaluation Metric — ½ p **(requirement (g))**
Explicit arithmetic, not prose:

```
WER_paper (exp_009, CV24 test)  = 24.89 %
WER_ours  (E2,      CV24 test)  = <X.XX> %
Absolute difference             = <X.XX> - 24.89 = <±D.DD> pp
Relative difference             = 100 x (<X.XX> - 24.89) / 24.89 = <±R.R> %
```

Repeat for CER (paper 8.075%) and for zero-shot WER (paper 120.4%). State the sign convention.

## 14. Visualizations — 2 p **(requirement (g))**
The nine figures from `docs/05`, each with a caption naming the run tag and date. Keep Figure 2 (val loss + WER twin axis) and Figure 8 (WER vs training hours) — they carry the analysis.

## 15. Discussion — 2 p **(requirement (h))**
Did the method reproduce? Separate the two claims (E0 verifies the reported result; E2 tests the recipe at reduced compute). Gap decomposition, ranked, from `docs/05` §5.4, with E3a/E3e giving measured rather than asserted attributions. Error analysis against the paper's ~70–78% substitution share and word-final morphology finding. Whether the zero-shot script-failure phenomenon reproduced. What we learned about Whisper transfer learning for low-resource Arabic-script languages. Honest note on the normalisation ablation: how much of the paper's 24.89% may be orthographic variation rather than recognition error.

## 16. Limitations — 1 p **(requirement (h))**
Split into three groups, which is what makes this section read as analysis rather than excuses:
- **Of the original paper** (quote its own stated limitations): single LoRA configuration; 50-step hyperparameter grid; no dialect stratification; no latency/memory measurement; CV20 and CV24 arms not comparable and exp_008/009/010 not a controlled scale ablation; unverified Urdu intermediate checkpoint; plus the internal inconsistencies we found (missing Table 4, 244M vs 307M, dev=test=13,791, zero-shot N=13,643 vs 13,791, optimizer unstated).
- **Of its artefacts**: both dataset mirrors 401; no public training code.
- **Of our reproduction**: compute-limited training subset; LR, precision and batch deviations; re-implemented trainer; single seed per configuration; no dialect-stratified evaluation; greedy decoding only.

## 17. Reproducibility Statement — 1½ p
Verbatim from `docs/06`.

## 18. Conclusion — ½ p
Three or four sentences: what was reproduced, the measured difference, the verdict, and one concrete next step (e.g. the Tier D full-data run on a rented A100).

## 19. References — ½–1 p
IEEE style, from `docs/08`. Every URL with an access date. Nothing cited that is not read.

## 20. Appendix — 3–4 p
- A: Repository link and exact commit SHA
- B: Every command, verbatim, in order
- C: `results/environment.txt`, `pip_freeze.txt`, `gpu.txt`
- D: **Execution evidence** — timestamped screenshots of the Colab/Kaggle session showing GPU allocation, training progress with step/loss/WER, the final evaluation output, and `nvidia-smi`. Console logs as text too, since screenshots compress badly.
- E: `results/preprocessing_log.csv` in full
- F: `results/sample_predictions.csv` — reference, zero-shot hypothesis, fine-tuned hypothesis, per-utterance WER. **Pashto must render; embed Noto Naskh Arabic and check RTL in the exported PDF.**
- G: `config.yaml` in full
- H: Link to our checkpoint, if published

---

## Production notes

- **Write with Section 11 empty until the runs finish.** Everything except 11, 12, 13, 14 and the Discussion can be completed before training ends.
- **Pashto rendering is a real failure mode.** If writing in LaTeX, use XeLaTeX or LuaLaTeX with `fontspec` + `polyglossia` (`\setotherlanguage{pashto}`) and a Noto Arabic font; `pdflatex` will not render it. If writing in Word, set the complex-script font explicitly and export, don't print-to-PDF. **Check the exported PDF on a different machine before submitting.**
- **Zero fabricated cells.** A `TBD` left in a draft is recoverable; an invented number is not.
- **One PDF.** Name it `CS444_A2_BSCS7B_ShiftII_<Group>_PashtoWhisper.pdf`.
