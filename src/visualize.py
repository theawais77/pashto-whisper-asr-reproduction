"""Generate the nine report figures from on-disk artefacts only.

Nothing here invents a number. Paper values come from config.yaml's
`paper_reference` block (verified against arXiv:2604.06507); our values come
from results/metrics.csv and results/<tag>/training_log.csv. Paper marks are
hatched and muted, ours solid and saturated, so the distinction is visible in
every chart (see docs/05 Step 6).

    python src/visualize.py --config config.yaml --tag exp009_repro_tierB
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_config

PAPER_STYLE = dict(color="#9aa5b1", hatch="//", edgecolor="#4a5568", linewidth=1.2)
OURS_STYLE = dict(color="#2b6cb0", edgecolor="#1a365d", linewidth=1.2)
PAPER_LABEL = "Rahman (2026), reported"
OURS_LABEL = "This reproduction"
DPI = 300


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def annotate_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)


# -- Figures 1 & 2: training / validation curves ---------------------------

def fig_loss_curves(log: pd.DataFrame, best_step, out_dir: Path) -> None:
    train = log.dropna(subset=["train_loss"]) if "train_loss" in log else pd.DataFrame()
    if not train.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train["step"], train["train_loss"], color="#2b6cb0", lw=1.4)
        if best_step:
            ax.axvline(best_step, ls="--", color="#c53030", lw=1,
                       label=f"best checkpoint (step {best_step})")
            ax.legend(frameon=False)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Training loss (cross-entropy)")
        ax.set_title("Training loss")
        ax.grid(alpha=0.3)
        save(fig, out_dir, "training_loss")

    evals = log.dropna(subset=["eval_loss"]) if "eval_loss" in log else pd.DataFrame()
    if not evals.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(evals["step"], evals["eval_loss"], color="#2b6cb0", marker="o",
                ms=4, lw=1.4, label="Validation loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation loss")
        if "eval_wer" in evals:
            twin = ax.twinx()
            twin.plot(evals["step"], evals["eval_wer"], color="#c53030",
                      marker="s", ms=4, lw=1.4, label="Validation WER")
            twin.set_ylabel("Validation WER (%)", color="#c53030")
            twin.tick_params(axis="y", colors="#c53030")
            lines = ax.get_lines() + twin.get_lines()
            ax.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="best")
        if best_step:
            ax.axvline(best_step, ls="--", color="#718096", lw=1)
        # The paper selects on WER, not loss, because its own Table 2 shows the
        # two are non-monotonic across runs. If our curves decouple too, that
        # reproduces a structural property of the paper -- discuss it.
        ax.set_title("Validation loss and WER (checkpoint selected on WER)")
        ax.grid(alpha=0.3)
        save(fig, out_dir, "validation_loss")


# -- Figure 3: zero-shot vs fine-tuned ------------------------------------

def fig_zeroshot_vs_finetuned(metrics: pd.DataFrame, out_dir: Path) -> None:
    zs = metrics[metrics["experiment"] == "zero-shot"]
    ft = metrics[metrics["experiment"] == "fine-tuned (ours)"]
    if zs.empty or ft.empty:
        return
    values = [zs["wer"].iloc[-1], ft["wer"].iloc[-1]]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["Zero-shot", "Fine-tuned"], values, width=0.55, **OURS_STYLE)
    bars[0].set_color("#c53030")
    bars[0].set_edgecolor("#742a2a")
    annotate_bars(ax, bars, "{:.2f}%")
    ax.axhline(100, ls=":", color="#718096", lw=1.2)
    ax.text(1.45, 102, "WER = 100%", fontsize=8, color="#718096", ha="right")
    ax.annotate(f"-{values[0] - values[1]:.1f} pp",
                xy=(0.5, max(values) * 0.55), ha="center", fontsize=11,
                color="#1a365d", weight="bold")
    ax.set_ylabel("WER (%)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title("Pashto WER: zero-shot vs fine-tuned whisper-small\n"
                 "Common Voice v24 test split")
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir, "wer_zeroshot_vs_finetuned")


# -- Figures 4 & 5: ours vs paper -----------------------------------------

def fig_comparison(metrics: pd.DataFrame, paper: dict, out_dir: Path,
                   metric: str, name: str, title: str, unit: str) -> None:
    zs = metrics[metrics["experiment"] == "zero-shot"]
    ft = metrics[metrics["experiment"] == "fine-tuned (ours)"]
    if ft.empty:
        return

    # One panel per configuration, each with its own y-axis. A single shared
    # axis is dominated by the ~120% zero-shot bars and squashes the fine-tuned
    # comparison -- the one the report actually turns on -- into illegibility.
    # Every panel still starts at zero, so no gap is visually exaggerated.
    panels = []
    if not zs.empty and metric == "wer":
        panels.append(("Zero-shot whisper-small",
                       paper["exp_009_small"]["zeroshot_wer"], zs["wer"].iloc[-1]))
    panels.append(("Fine-tuned whisper-small",
                   paper["exp_009_small"]["wer_norm" if metric == "wer" else "cer"],
                   ft[metric].iloc[-1]))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6))
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (panel_title, theirs, ours) in zip(axes, panels):
        ax.bar([0], [theirs], 0.5, label=PAPER_LABEL, **PAPER_STYLE)
        ax.bar([1], [ours], 0.5, label=OURS_LABEL, **OURS_STYLE)
        for x, v in ((0, theirs), (1, ours)):
            ax.annotate(f"{v:.2f}", (x, v), ha="center", va="bottom", fontsize=10,
                        xytext=(0, 3), textcoords="offset points")
        top = max(theirs, ours) * 1.38
        ax.set_ylim(0, top)                      # never truncate the axis
        # position the delta at a fixed fraction of the axis, never relative to
        # bar height, or it collides with the value labels on short bars
        ax.annotate(f"{ours - theirs:+.2f} pp\n({100 * (ours - theirs) / theirs:+.1f} %)",
                    (0.5, top * 0.88), ha="center", va="top",
                    fontsize=10, color="#742a2a", weight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["paper", "ours"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(unit)
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(title, y=1.04)
    save(fig, out_dir, name)


# -- Figure 6: error-type breakdown ---------------------------------------

def fig_error_breakdown(metrics: pd.DataFrame, out_dir: Path) -> None:
    rows = metrics[metrics["experiment"].isin(["zero-shot", "fine-tuned (ours)"])]
    if rows.empty:
        return
    labels = list(rows["experiment"])
    bottom = [0.0] * len(rows)
    colours = {"sub_pct_of_errors": "#2b6cb0", "del_pct_of_errors": "#d69e2e",
               "ins_pct_of_errors": "#c53030"}
    names = {"sub_pct_of_errors": "Substitutions", "del_pct_of_errors": "Deletions",
             "ins_pct_of_errors": "Insertions"}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for column, colour in colours.items():
        values = list(rows[column].fillna(0))
        ax.bar(labels, values, bottom=bottom, label=names[column],
               color=colour, edgecolor="white", width=0.5)
        bottom = [b + v for b, v in zip(bottom, values)]
    # Paper's Appendix C: substitutions are ~70-78% of word errors.
    ax.axhline(74, ls="--", color="#4a5568", lw=1)
    ax.text(len(labels) - 0.5, 75.5, "paper: ~70-78% substitutions",
            fontsize=8, ha="right", color="#4a5568")
    ax.set_ylabel("Share of total word errors (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Error-type breakdown")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir, "error_breakdown")


# -- Figure 8: WER vs training hours --------------------------------------

def fig_wer_vs_hours(metrics: pd.DataFrame, paper: dict, out_dir: Path) -> None:
    rows = metrics[(metrics["experiment"] == "fine-tuned (ours)")
                   & metrics["train_hours"].notna()]
    if len(rows) < 2:
        return
    rows = rows.sort_values("train_hours")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(rows["train_hours"], rows["wer"], marker="o", color="#2b6cb0",
            lw=1.6, label=OURS_LABEL)
    ax.scatter([paper["splits"]["train_plus_dev_hours"]],
               [paper["exp_009_small"]["wer_norm"]],
               marker="*", s=260, color="#c53030", zorder=5,
               label=f"{PAPER_LABEL} (113.4 h, 24.89%)")
    ax.set_xscale("log")
    ax.set_xlabel("Training data (hours, log scale)")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER vs training data volume\nquantifies the compute-induced gap")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    save(fig, out_dir, "wer_vs_hours")


# -- Figure 9: reproduce the paper's model-scale figure -------------------

def fig_model_scale(metrics: pd.DataFrame, paper: dict, out_dir: Path) -> None:
    names = ["whisper-base\n(74M)", "whisper-small\n(244M)", "turbo\n(809M)"]
    reported = [paper["exp_008_base"]["wer_norm"],
                paper["exp_009_small"]["wer_norm"],
                paper["exp_010_turbo"]["wer_norm"]]

    ours_by_model = {}
    for _, row in metrics[metrics["experiment"] == "fine-tuned (ours)"].iterrows():
        model = str(row["model"])
        index = 0 if "base" in model else 2 if ("turbo" in model or "large" in model) else 1
        ours_by_model[index] = row["wer"]

    positions = range(3)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_p = ax.bar([p - width / 2 for p in positions], reported, width,
                    label=PAPER_LABEL, **PAPER_STYLE)
    annotate_bars(ax, bars_p, "{:.2f}")
    if ours_by_model:
        bars_o = ax.bar([p + width / 2 for p in ours_by_model], list(ours_by_model.values()),
                        width, label=OURS_LABEL, **OURS_STYLE)
        annotate_bars(ax, bars_o, "{:.2f}")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(names)
    ax.set_ylabel("WER (%)")
    ax.set_ylim(0, max(reported + list(ours_by_model.values()) or reported) * 1.25)
    ax.set_title("Model scale on Common Voice v24 Pashto test\n"
                 "paper's Table 2, with our measured point overlaid")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir, "fig_strategy_or_scale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tag", default=None, help="run whose curves to plot")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg["paths"]["results"])
    paper = cfg["paper_reference"]

    metrics_path = results_dir / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} missing -- run src/evaluate.py first")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["source"] == "ours"].drop_duplicates(
        subset=["tag"], keep="last"
    )

    print("generating figures:")
    if args.tag:
        run_dir = results_dir / args.tag
        log_path = run_dir / "training_log.csv"
        best_step = None
        best_path = run_dir / "best_checkpoint.json"
        if best_path.exists():
            import json
            best_step = json.loads(best_path.read_text()).get("best_step")
        if log_path.exists():
            fig_loss_curves(pd.read_csv(log_path), best_step, results_dir)
        else:
            print(f"  (skipping curves: {log_path} missing)")

    fig_zeroshot_vs_finetuned(metrics, results_dir)
    fig_comparison(metrics, paper, results_dir, "wer", "wer_comparison",
                   "WER: our reproduction vs Rahman (2026)\n"
                   "whisper-small, Common Voice v24 Pashto test", "WER (%)")
    fig_comparison(metrics, paper, results_dir, "cer", "cer_comparison",
                   "CER: our reproduction vs Rahman (2026)", "CER (%)")
    fig_error_breakdown(metrics, results_dir)
    fig_wer_vs_hours(metrics, paper, results_dir)
    fig_model_scale(metrics, paper, results_dir)

    # Report-ready comparison table (Step 13 arithmetic, no prose).
    finetuned = metrics[metrics["experiment"] == "fine-tuned (ours)"]
    if not finetuned.empty:
        row = finetuned.iloc[-1]
        paper_wer = paper["exp_009_small"]["wer_norm"]
        paper_cer = paper["exp_009_small"]["cer"]
        lines = [
            "Metric,Paper (exp_009),Ours,Absolute difference (pp),Relative difference (%)",
            f"WER (normalised),{paper_wer},{row['wer']:.2f},"
            f"{row['wer'] - paper_wer:+.2f},{100 * (row['wer'] - paper_wer) / paper_wer:+.1f}",
            f"CER,{paper_cer},{row['cer']:.2f},"
            f"{row['cer'] - paper_cer:+.2f},{100 * (row['cer'] - paper_cer) / paper_cer:+.1f}",
        ]
        (results_dir / "comparison_table.csv").write_text("\n".join(lines), encoding="utf-8")
        print("  comparison_table.csv")
        print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
