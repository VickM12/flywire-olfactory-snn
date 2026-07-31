"""
generate_figures2_3.py
----------------------
Generates Figure 2 (stopped epoch distribution strip plot) and
Figure 3 (validation accuracy learning curves) for the paper.

Figure 2: Per-run stopped epoch distributions showing ConnectomeSNN
          consistently trains longer than ShuffledSNN.

Figure 3: Mean val accuracy vs epoch for ConnectomeSNN vs ShuffledSNN
          with ±1 std shading, reconstructed from per-epoch log data.

Usage:
    python generate_figures2_3.py
    python generate_figures2_3.py --results path/to/comparison.json
                                  --log path/to/run.log
                                  --out figures/
"""

import argparse
import json
import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

COLORS = {
    "ConnectomeSNN": "#534AB7",
    "ShuffledSNN":   "#9F99E0",
    "SparseMLP":     "#3B8BD4",
    "DenseMLP":      "#1D9E75",
}

MODELS = ["ConnectomeSNN", "ShuffledSNN", "SparseMLP", "DenseMLP"]


# ── Figure 2: Strip plot of stopped epochs ───────────────────────────────────

def make_figure2(results_path: str, out_stem: str) -> None:
    with open(results_path) as f:
        data = json.load(f)

    per_run = data["per_run"]

    by_model = defaultdict(list)
    for r in per_run:
        by_model[r["model"]].append(r["stopped_epoch"])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    np.random.seed(42)
    for i, model in enumerate(MODELS):
        epochs = np.array(by_model[model])
        jitter = np.random.uniform(-0.12, 0.12, size=len(epochs))

        # individual points
        ax.scatter(np.full_like(epochs, i, dtype=float) + jitter,
                   epochs,
                   color=COLORS[model],
                   s=48, alpha=0.75, zorder=3,
                   label=model)

        # mean line
        mean = np.mean(epochs)
        ax.hlines(mean, i - 0.28, i + 0.28,
                  colors=COLORS[model],
                  linewidths=2.5, zorder=4)

        # std range
        std = np.std(epochs)
        ax.vlines(i, mean - std, mean + std,
                  colors=COLORS[model],
                  linewidths=1.2, alpha=0.5, zorder=3)

        # annotate mean
        ax.text(i, mean + std + 1.0,
                f"{mean:.1f}",
                ha="center", va="bottom",
                fontsize=9, color=COLORS[model], fontweight="bold")

    # sign test annotation for ConnectomeSNN vs ShuffledSNN
    conn = np.array(by_model["ConnectomeSNN"])
    shuf = np.array(by_model["ShuffledSNN"])
    n_conn_longer = np.sum(conn > shuf)
    n = len(conn)
    p_val = stats.binomtest(n_conn_longer, n, 0.5, alternative="greater").pvalue
    ax.annotate(
        f"ConnectomeSNN > ShuffledSNN\nin {n_conn_longer}/{n} runs\n(sign test p = {p_val:.4f})",
        xy=(0.5, 0.97), xycoords="axes fraction",
        ha="center", va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec="#cccccc", alpha=0.9)
    )

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(["Connectome\nSNN", "Shuffled\nSNN",
                        "Sparse\nMLP", "Dense\nMLP"], fontsize=10)
    ax.set_ylabel("Stopped epoch", fontsize=11)
    ax.set_ylim(0, 58)
    ax.set_title("(a) Training duration — per-run distribution", fontsize=11, pad=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(out_stem, exist_ok=True)
    fig.savefig(os.path.join(out_stem, "figure2.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_stem, "figure2.png"), dpi=300, bbox_inches="tight")
    print(f"Saved figure2.pdf / figure2.png")
    plt.close(fig)


# ── Figure 3: Learning curves from log ───────────────────────────────────────

def parse_log(log_path: str) -> dict:
    """
    Parse run.log and extract per-epoch val_acc for ConnectomeSNN and ShuffledSNN.
    Returns dict: model -> list of val_acc arrays (one per run).
    """
    pattern = re.compile(
        r"\[DoOR/(ConnectomeSNN|ShuffledSNN)\] epoch (\d+)/\d+ "
        r"loss=[\d.]+ train_acc=[\d.]+ val_acc=([\d.]+)"
    )

    # collect: model -> run_id -> epoch -> val_acc
    runs = defaultdict(lambda: defaultdict(dict))
    run_counters = defaultdict(int)
    current_run = {}   # model -> current run id

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                model = m.group(1)
                epoch = int(m.group(2))
                val_acc = float(m.group(3))

                # detect new run: epoch 1 starts a new run
                if epoch == 1:
                    run_id = run_counters[model]
                    run_counters[model] += 1
                    current_run[model] = run_id

                rid = current_run.get(model, 0)
                runs[model][rid][epoch] = val_acc

    # convert to list of arrays padded to max_epoch per run
    result = {}
    for model, model_runs in runs.items():
        arrays = []
        for rid, epoch_dict in model_runs.items():
            max_ep = max(epoch_dict.keys())
            arr = np.full(max_ep, np.nan)
            for ep, val in epoch_dict.items():
                arr[ep - 1] = val
            arrays.append(arr)
        result[model] = arrays

    return result


def make_figure3(log_path: str, out_stem: str) -> None:
    curves = parse_log(log_path)

    snn_models = ["ConnectomeSNN", "ShuffledSNN"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, model in zip(axes, snn_models):
        arrays = curves.get(model, [])
        if not arrays:
            ax.set_title(f"{model} — no data")
            continue

        max_len = max(len(a) for a in arrays)
        padded = np.full((len(arrays), max_len), np.nan)
        for i, a in enumerate(arrays):
            padded[i, :len(a)] = a

        # count non-nan per epoch for mean/std
        mean_curve = np.nanmean(padded, axis=0)
        std_curve  = np.nanstd(padded, axis=0)
        epochs     = np.arange(1, max_len + 1)

        color = COLORS[model]

        # individual run traces (faint)
        for run_arr in arrays:
            run_epochs = np.arange(1, len(run_arr) + 1)
            ax.plot(run_epochs, run_arr,
                    color=color, alpha=0.12, linewidth=0.8)

        # mean ± std
        ax.fill_between(epochs,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=color, alpha=0.18)
        ax.plot(epochs, mean_curve,
                color=color, linewidth=2.2,
                label=f"Mean ({len(arrays)} runs)")

        # annotate final mean
        last_valid = np.where(~np.isnan(mean_curve))[0]
        if len(last_valid):
            last_ep = last_valid[-1]
            ax.annotate(f"{mean_curve[last_ep]:.3f}",
                        xy=(epochs[last_ep], mean_curve[last_ep]),
                        xytext=(epochs[last_ep] - 3, mean_curve[last_ep] + 0.015),
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="->",
                                        color=color, lw=0.8))

        label = "Connectome SNN" if model == "ConnectomeSNN" else "Shuffled SNN"
        ax.set_title(f"({chr(97 + snn_models.index(model))}) {label}",
                     fontsize=11, pad=6)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_xlim(1, max_len + 1)
        ax.set_ylim(0.05, 0.56)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, frameon=False)

    axes[0].set_ylabel("Testing accuracy", fontsize=10)

    fig.suptitle("Testing accuracy learning curves — SNN models",
                 fontsize=11, y=1.01)

    plt.tight_layout()
    os.makedirs(out_stem, exist_ok=True)
    fig.savefig(os.path.join(out_stem, "figure3.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_stem, "figure3.png"), dpi=300, bbox_inches="tight")
    print(f"Saved figure3.pdf / figure3.png")
    plt.close(fig)


# ── Sign test report ──────────────────────────────────────────────────────────

def report_sign_test(results_path: str) -> None:
    with open(results_path) as f:
        data = json.load(f)

    per_run = data["per_run"]
    by_model = defaultdict(list)
    for r in per_run:
        by_model[r["model"]].append(r["stopped_epoch"])

    conn = np.array(by_model["ConnectomeSNN"])
    shuf = np.array(by_model["ShuffledSNN"])

    n_longer = np.sum(conn > shuf)
    n = len(conn)
    p = stats.binomtest(n_longer, n, 0.5, alternative="greater").pvalue

    print(f"\nSign test: ConnectomeSNN stopped later than ShuffledSNN")
    print(f"  Runs where ConnectomeSNN > ShuffledSNN: {n_longer}/{n}")
    print(f"  One-sided p-value: {p:.4f}")
    print(f"\nPer-run comparison:")
    for i, (c, s) in enumerate(zip(conn, shuf)):
        marker = "✓" if c > s else "✗"
        print(f"  Run {i+1:2d}: Connectome={c:2.0f}  Shuffled={s:2.0f}  {marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/comparison-2026-03-25.json")
    parser.add_argument("--log",     default="results/run.log")
    parser.add_argument("--out",     default="figures")
    args = parser.parse_args()

    report_sign_test(args.results)
    make_figure2(args.results, args.out)
    make_figure3(args.log, args.out)


if __name__ == "__main__":
    main()