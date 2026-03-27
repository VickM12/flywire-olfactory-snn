"""
generate_figure3_all.py
-----------------------
Generates Figure 3: learning curves for all four model conditions
on a single plot, showing mean val accuracy ± 1 std across 15 runs.

Usage:
    python generate_figure3_all.py
    python generate_figure3_all.py --log results/run.log --out figures/
"""

import argparse
import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

COLORS = {
    "ConnectomeSNN": "#534AB7",
    "ShuffledSNN":   "#9F99E0",
    "SparseMLP":     "#3B8BD4",
    "DenseMLP":      "#1D9E75",
}

LABELS = {
    "ConnectomeSNN": "ConnectomeSNN",
    "ShuffledSNN":   "ShuffledSNN",
    "SparseMLP":     "SparseMLP",
    "DenseMLP":      "DenseMLP",
}

MODELS = ["DenseMLP", "SparseMLP", "ConnectomeSNN", "ShuffledSNN"]


def parse_log(log_path: str) -> dict:
    pattern = re.compile(
        r"\[DoOR/(ConnectomeSNN|ShuffledSNN|SparseMLP|DenseMLP)\] epoch (\d+)/\d+ "
        r"loss=[\d.]+ train_acc=[\d.]+ val_acc=([\d.]+)"
    )

    runs = defaultdict(lambda: defaultdict(dict))
    run_counters = defaultdict(int)
    current_run = {}

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                model = m.group(1)
                epoch = int(m.group(2))
                val_acc = float(m.group(3))
                if epoch == 1:
                    run_id = run_counters[model]
                    run_counters[model] += 1
                    current_run[model] = run_id
                rid = current_run.get(model, 0)
                runs[model][rid][epoch] = val_acc

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


def make_figure3_all(log_path: str, out_stem: str) -> None:
    curves = parse_log(log_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODELS:
        arrays = curves.get(model, [])
        if not arrays:
            continue

        max_len = max(len(a) for a in arrays)
        padded = np.full((len(arrays), max_len), np.nan)
        for i, a in enumerate(arrays):
            padded[i, :len(a)] = a

        mean_curve = np.nanmean(padded, axis=0)
        std_curve  = np.nanstd(padded,  axis=0)
        epochs     = np.arange(1, max_len + 1)
        color      = COLORS[model]

        # shaded std band
        ax.fill_between(epochs,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=color, alpha=0.13)

        # mean line
        ax.plot(epochs, mean_curve,
                color=color, linewidth=2.0,
                label=LABELS[model])

        # terminal dot at last valid point
        last_idx = np.where(~np.isnan(mean_curve))[0]
        if len(last_idx):
            li = last_idx[-1]
            ax.scatter(epochs[li], mean_curve[li],
                       color=color, s=40, zorder=5)

    # visual separator between early fast phase and slower phase
    ax.axvline(x=7, color="#cccccc", linewidth=0.8,
               linestyle="--", alpha=0.7)
    ax.text(7.3, 0.12, "epoch 7", fontsize=7.5,
            color="#aaaaaa", va="bottom")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation accuracy", fontsize=11)
    ax.set_xlim(1, 50)
    ax.set_ylim(0.05, 0.58)
    ax.set_title("Validation accuracy — all models (mean ± 1 std, 15 runs each)",
                 fontsize=11, pad=8)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # legend — ordered by final performance
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              fontsize=9.5, frameon=False,
              loc="lower right")

    # annotations for key observations
    ax.annotate("MLPs converge\nby ~epoch 15",
                xy=(15, 0.49), xytext=(18, 0.44),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->",
                                color="#aaaaaa", lw=0.8))

    ax.annotate("SNNs continue\nlearning to ~epoch 27",
                xy=(27, 0.47), xytext=(30, 0.38),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->",
                                color="#aaaaaa", lw=0.8))

    plt.tight_layout()
    os.makedirs(out_stem, exist_ok=True)
    fig.savefig(os.path.join(out_stem, "figure3_all.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_stem, "figure3_all.png"),
                dpi=300, bbox_inches="tight")
    print(f"Saved figure3_all.pdf / figure3_all.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="results/run.log")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    make_figure3_all(args.log, args.out)


if __name__ == "__main__":
    main()