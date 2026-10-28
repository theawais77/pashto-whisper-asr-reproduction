"""Render the methodology diagram for report sections 5 and 9.

Generated rather than hand-drawn so it regenerates from the repository and
stays consistent with the other figures' palette.

    python src/make_methodology_figure.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# palette shared with src/visualize.py
BLUE, RED, GREY, AMBER = "#2b6cb0", "#c53030", "#4a5568", "#b7791f"
PAPER_FILL, PAPER_EDGE = "#e2e8f0", "#718096"
OURS_FILL, OURS_EDGE = "#dbeafe", "#2b6cb0"
DATA_FILL, DATA_EDGE = "#e6fffa", "#2c7a7b"
EVAL_FILL, EVAL_EDGE = "#fefcbf", "#b7791f"
OUT_FILL, OUT_EDGE = "#f0fff4", "#276749"

fig, ax = plt.subplots(figsize=(11.5, 15))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.invert_yaxis()
ax.axis("off")


def box(x, y, w, h, title, body="", fill="#ffffff", edge=GREY,
        title_size=9.5, body_size=8, bold_title=True, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
        linewidth=1.3, facecolor=fill, edgecolor=edge, linestyle=ls, zorder=2))
    if body:
        ax.text(x + w / 2, y + h * 0.30, title, ha="center", va="center",
                fontsize=title_size, weight="bold" if bold_title else "normal",
                color=edge, zorder=3)
        ax.text(x + w / 2, y + h * 0.68, body, ha="center", va="center",
                fontsize=body_size, color="#1a202c", zorder=3, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, weight="bold" if bold_title else "normal",
                color=edge, zorder=3, linespacing=1.45)
    return (x + w / 2, y, x + w / 2, y + h)        # top-centre, bottom-centre


def arrow(p_from, p_to, color=GREY, ls="-", lw=1.5, label="", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=color, linestyle=ls, zorder=1,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=4))
    if label:
        ax.text((p_from[0] + p_to[0]) / 2 + 1.2, (p_from[1] + p_to[1]) / 2,
                label, fontsize=7.2, color=color, ha="left", va="center",
                style="italic", zorder=3)


def band(y, text):
    ax.text(1.5, y, text, fontsize=10.5, weight="bold", color="#1a202c",
            ha="left", va="center")
    ax.plot([1.5, 98.5], [y + 1.6, y + 1.6], color="#cbd5e0", lw=1, zorder=0)


# ===================================================================== STAGE 1
band(2.0, "1 · Base paper and its released artefacts")

b_paper = box(6, 5, 88, 5.2,
              "Rahman (2026) — “Fine-tuning Whisper for Pashto ASR: strategies and scale”",
              "arXiv:2604.06507 · DOI 10.48550/arXiv.2604.06507 · 7 April 2026 · preprint, cs.CL",
              PAPER_FILL, PAPER_EDGE, 10)

b_target = box(6, 13, 42, 6.6,
               "Reproduction target — exp_009",
               "whisper-small, full fine-tuning, Common Voice v24\n"
               "Reported: WER 24.89 % · CER 8.075 %\n"
               "Zero-shot baseline reported: 120.4 % WER",
               PAPER_FILL, PAPER_EDGE)

b_art = box(52, 13, 42, 6.6,
            "Released artefacts (Apache-2.0)",
            "ihanif/exp_009_small_cv24_vanilla\n"
            "experiment_log.txt · training_args.bin\n"
            "Training code and corpora: NOT public (HTTP 401)",
            PAPER_FILL, PAPER_EDGE)

b_rec = box(52, 22.5, 42, 6.2,
            "Configuration recovered by us",
            "lr 1e-4 · true batch 64 · FP32 · warmup 500\n"
            "weight decay 0.0 · patience 5 · 0.5–30 s\n"
            "eval_dataset_splits = TEST  (selection on test)",
            "#fff5f5", RED)

arrow((b_paper[0] - 22, b_paper[3]), (b_target[0], b_target[1]))
arrow((b_paper[0] + 22, b_paper[3]), (b_art[0], b_art[1]))
arrow((b_art[0], b_art[3]), (b_rec[0], b_rec[1]), RED, label="parsed")

# ===================================================================== STAGE 2
band(32.5, "2 · Data preparation")

b_cv = box(6, 35.5, 88, 4.6,
           "Mozilla Common Voice Scripted Speech 24.0 — Pashto (ps), CC0-1.0",
           "Official HF mirrors empty since Oct 2025 → obtained from Mozilla Data Collective / CC0 Kaggle mirror",
           DATA_FILL, DATA_EDGE, 10, 7.8)

b_split = box(6, 43, 42, 5.6,
              "Official shipped splits — no re-split",
              "train.tsv / dev.tsv / test.tsv\n"
              "speaker-disjoint by construction; verified",
              DATA_FILL, DATA_EDGE)

b_filt = box(52, 43, 42, 5.6,
             "Filters (paper + recovered config)",
             "non-empty · Arabic script · de-duplicated\n"
             "≤ 448 label tokens · 0.5–30 s · 16 kHz mono",
             DATA_FILL, DATA_EDGE)

b_tr = box(6, 52, 27, 5.4, "Train (subset)",
           "~12 h seeded, seed 42\npaper used 113.4 h", DATA_FILL, DATA_EDGE, 9, 7.6)
b_dv = box(36.5, 52, 27, 5.4, "Dev — held OUT",
           "1,000 utterances\npaper merged dev into train", "#fff5f5", RED, 9, 7.6)
b_te = box(67, 52, 27, 5.4, "Test — untouched",
           "13,791 utterances / 20.7 h\nused once, at the end", DATA_FILL, DATA_EDGE, 9, 7.6)

arrow((b_cv[0] - 22, b_cv[3]), (b_split[0], b_split[1]))
arrow((b_cv[0] + 22, b_cv[3]), (b_filt[0], b_filt[1]))
for src, dst in ((b_split, b_tr), (b_filt, b_dv), (b_filt, b_te)):
    arrow((src[0], src[3]), (dst[0], dst[1]))

# ===================================================================== STAGE 3
band(60.5, "3 · Model and the three experimental arms")

b_model = box(6, 63.5, 88, 4.4,
              "openai/whisper-small — 244 M parameters, encoder–decoder Transformer",
              "80-channel log-Mel, 30 s window → 12 encoder layers → 12 decoder layers → 51,865 BPE  "
              "· prompt ⟨|ps|⟩ ⟨|transcribe|⟩",
              "#ffffff", GREY, 10, 7.8)

b_b0 = box(6, 70.5, 27, 7.2, "B0 · Zero-shot",
           "untouched weights\nno training\n\ntarget: >100 % WER\npaper: 120.4 %",
           PAPER_FILL, PAPER_EDGE, 9.5, 7.6)
b_b = box(36.5, 70.5, 27, 7.2, "B · Author checkpoint",
          "his weights, OUR harness\nno training\n\ntarget: 24.89 % WER\n8.075 % CER",
          "#fff5f5", RED, 9.5, 7.6)
b_c = box(67, 70.5, 27, 7.2, "C · Our fine-tuning",
          "full FT · AdamW · lr 1e-5\neff. batch 32 · fp16\ngrad-checkpointing\nbest on val WER, patience 5",
          OURS_FILL, OURS_EDGE, 9.5, 7.6)

for dst in (b_b0, b_b, b_c):
    arrow((b_model[0] + (dst[0] - b_model[0]) * 0.92, b_model[3]), (dst[0], dst[1]))
arrow((b_tr[0], b_tr[3]), (b_c[0] - 6, b_c[1]), OURS_EDGE, rad=-0.12, label="train")
arrow((b_dv[0], b_dv[3]), (b_c[0] - 10, b_c[1]), OURS_EDGE, rad=-0.05, label="select")

# ===================================================================== STAGE 4
band(80.0, "4 · Evaluation, our own analyses, and comparison")

b_dec = box(6, 82.5, 88, 3.6,
            "Identical decoding for all three arms — greedy, max_new_tokens 225, language = ps",
            "", EVAL_FILL, EVAL_EDGE, 9.5)
arrow((b_te[0], b_te[3]), (b_dec[0] + 30, b_dec[1]), DATA_EDGE, rad=0.10, label="test audio")
for src in (b_b0, b_b, b_c):
    arrow((src[0], src[3]), (src[0], b_dec[1]))

b_met = box(6, 88, 42, 5.4, "Metrics",
            "BasicTextNormalizer (paper's normaliser)\n"
            "WER_norm ← headline · WER_ortho · CER\n"
            "bootstrap 95 % CI · S/D/I breakdown",
            EVAL_FILL, EVAL_EDGE, 9.5, 7.6)

b_ours = box(52, 88, 42, 5.4, "D · Our own analyses (not in the paper)",
             "D1 Pashto-script output rate\n"
             "D2 orthographic-normalisation ablation\n"
             "paired bootstrap, zero-shot vs fine-tuned",
             OURS_FILL, OURS_EDGE, 9.5, 7.6)

arrow((b_dec[0] - 22, b_dec[3]), (b_met[0], b_met[1]), EVAL_EDGE)
arrow((b_dec[0] + 22, b_dec[3]), (b_ours[0], b_ours[1]), OURS_EDGE)

b_cmp = box(6, 96.2, 88, 3.4,
            "Difference vs the paper  =  ours − 24.89 %   (percentage points and relative)"
            "   →  Figures 1–8, metrics.csv, report §§6–9",
            "", OUT_FILL, OUT_EDGE, 9.5)
arrow((b_met[0], b_met[3]), (b_cmp[0] - 22, b_cmp[1]), EVAL_EDGE)
arrow((b_ours[0], b_ours[3]), (b_cmp[0] + 22, b_cmp[1]), OURS_EDGE)

# ------------------------------------------------- the two-claims callout
ax.add_patch(FancyBboxPatch(
    (6, 77.0), 88, 4.0, boxstyle="round,pad=0.3,rounding_size=0.6",
    linewidth=1.3, facecolor="#fffaf0", edgecolor=AMBER,
    linestyle=(0, (4, 2)), zorder=2))
ax.text(50, 79.0,
        "Why arms B and C are separate:  B tests whether the paper's REPORTED NUMBER is verifiable "
        "(measurement).\n"
        "C tests whether its TRAINING RECIPE reproduces at our compute budget. Different claims, "
        "different answers.",
        ha="center", va="center", fontsize=8.2, color="#744210",
        linespacing=1.5, zorder=3)

fig.suptitle("Reproduction methodology — fine-tuning Whisper for Pashto ASR\n"
             "CS-444 Assignment 2 · reproduction of Rahman (2026), arXiv:2604.06507",
             fontsize=13, y=0.993)

out = pathlib.Path("results")
out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"methodology.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print("wrote results/methodology.png and results/methodology.pdf")
