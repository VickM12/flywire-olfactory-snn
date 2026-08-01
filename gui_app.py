from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flywire_snn.config import ALL_MODEL_NAMES, ExperimentConfig
from flywire_snn.models.dense_mlp import DenseMLP
from flywire_snn.models.shuffled_snn import ShuffledSNN
from flywire_snn.models.sparse_mlp import SparseMLP, recurrent_sparsity_ratio
from flywire_snn.models.snn import MaskedRecurrentLIFSNN

RUNNER = ROOT / "run_experiment.py"
CONNECTOME_NPZ = "olfactory_connectome.npz"
CONNECTOME_META = "olfactory_connectome.meta.json"
DOOR_CSV = "door_or_merged.csv"

# Third-party DEBUG spam at DEBUG level (keep INFO/WARNING/ERROR from any module)
_NOISY_DEBUG_MARKERS = (
    "python_jsonschema_objects",
    "jsonschema",
    "urllib3",
    "asyncio",
    "PIL.",
    "matplotlib",
)


def _latest_comparison_json(result_dir: Path) -> Optional[Path]:
    if not result_dir.exists():
        return None
    files = sorted(result_dir.glob("comparison-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_log_tail(path: Path, max_bytes: int = 400_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size <= max_bytes:
            f.seek(0)
            return f.read().decode("utf-8", errors="replace")
        f.seek(max(0, size - max_bytes))
        data = f.read()
    nl = data.find(b"\n")
    if nl != -1:
        data = data[nl + 1 :]
    return data.decode("utf-8", errors="replace")


def _keep_line(line: str, hide_noise: bool) -> bool:
    if not hide_noise:
        return True
    if "| DEBUG |" not in line:
        return True
    return not any(m in line for m in _NOISY_DEBUG_MARKERS)


def _compact_line(line: str) -> str:
    parts = line.split(" | ")
    if len(parts) < 4:
        return line
    ts, level, name, msg = parts[0], parts[1], parts[2], " | ".join(parts[3:])
    time_only = ts.split()[-1] if " " in ts else ts[-8:]
    short = name.split(".")[-1] if "." in name else name
    if len(short) > 30:
        short = short[:27] + "..."
    return f"{time_only}  {level:5}  {short:30}  {msg}"


def _format_log(raw: str, max_lines: int, hide_noise: bool, compact: bool) -> str:
    lines = raw.splitlines()
    lines = [ln for ln in lines if _keep_line(ln, hide_noise)]
    if compact:
        lines = [_compact_line(ln) for ln in lines]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def _render_summary(payload: Dict[str, Any]) -> None:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    if not isinstance(summary, dict) or not summary:
        st.info("No `summary` found in this JSON.")
        return
    for ds_name, ds_summ in summary.items():
        if not isinstance(ds_summ, dict):
            continue
        st.subheader(ds_name)
        rows = []
        order = ["ConnectomeSNN", "ShuffledSNN", "SparseMLP", "DenseMLP"]
        for model_name in order:
            if model_name not in ds_summ:
                continue
            m = ds_summ[model_name]
            rows.append(
                {
                    "Model": model_name,
                    "Test Acc (mean)": (m.get("test_acc") or {}).get("mean"),
                    "Test Acc (std)": (m.get("test_acc") or {}).get("std"),
                    "Epochs to 80% (mean)": (m.get("epochs_to_80") or {}).get("mean"),
                    "Stopped epoch (mean)": (m.get("stopped_epoch") or {}).get("mean"),
                    "Spike sparsity (mean)": (m.get("spike_sparsity") or {}).get("mean"),
                    "Params": m.get("params"),
                }
            )
        if rows:
            st.dataframe(rows, width="stretch")


def _start_subprocess(args: List[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, str(RUNNER), *args],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _door_dims(data_dir: Path) -> Optional[Tuple[int, int]]:
    """Return (num_classes, input_dim) from cached DoOR table, or None."""
    p = data_dir / "processed" / DOOR_CSV
    if not p.exists():
        return None
    df = pd.read_csv(p)
    rec_cols = [c for c in df.columns if c != "odor_key"]
    return int(len(df)), int(len(rec_cols))


def _load_connectome_meta(data_dir: Path) -> Optional[Dict[str, Any]]:
    mpath = data_dir / "processed" / CONNECTOME_META
    if not mpath.exists():
        return None
    with mpath.open("r", encoding="utf-8") as f:
        return json.load(f)


def _subsample_square(A: sp.csr_matrix, max_n: int, rng: np.random.Generator) -> np.ndarray:
    n = A.shape[0]
    if n <= max_n:
        return A.toarray()
    idx = np.sort(rng.choice(n, size=max_n, replace=False))
    sub = A[np.ix_(idx, idx)]
    return sub.toarray()


def _parameter_count_module(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _render_model_visualization(data_dir: Path) -> None:
    st.subheader("Connectome (cached graph)")
    cache = data_dir / "processed" / CONNECTOME_NPZ
    meta = _load_connectome_meta(data_dir)
    if meta:
        st.caption(
            f"Cache metadata: source={meta.get('source')}, neurons={meta.get('neurons')}, "
            f"edges={meta.get('edges')}"
            + (f", has_positions={meta.get('has_positions')}" if "has_positions" in meta else "")
        )

    if not cache.exists():
        st.warning(
            f"No connectome at `{cache}`. Run the experiment once (or fetch the graph via CLI) "
            "so `olfactory_connectome.npz` exists."
        )
    else:
        A = sp.load_npz(cache).tocsr()
        n = A.shape[0]
        nnz = A.nnz
        dens = nnz / max(n * n, 1)
        d = A.data
        n_pos = int(np.sum(d > 0)) if d.size else 0
        n_neg = int(np.sum(d < 0)) if d.size else 0
        positions_nm = None
        pos_cache = data_dir / "processed" / CONNECTOME_NPZ.replace(".npz", "_positions.npz")
        if pos_cache.exists():
            try:
                z = np.load(pos_cache)
                if "positions_nm" in z:
                    positions_nm = z["positions_nm"]
            except Exception:
                positions_nm = None
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Neurons (N)", f"{n}")
        c2.metric("Directed edges (nnz)", f"{nnz}")
        c3.metric("Recurrent density", f"{dens:.4%}")
        c4.metric("Sign + / −", f"{n_pos} / {n_neg}")
        if positions_nm is None:
            st.caption(f"Positions cache: missing (`{pos_cache.name}` not loaded). Wiring will be force-directed.")
        else:
            finite = np.isfinite(positions_nm[:, :3]).all(axis=1)
            st.caption(
                f"Positions cache: loaded `{pos_cache.name}` — "
                f"{int(finite.sum())}/{int(finite.size)} neurons have finite (x,y,z)."
            )

        rng_seed = st.number_input("Subsample RNG seed (heatmap)", min_value=0, max_value=999_999, value=42)
        _hi = max(1, min(512, n))
        _lo = min(64, _hi) if _hi >= 64 else 1
        _def = max(_lo, min(256, _hi))
        max_preview = st.slider("Heatmap: neuron subsample size", _lo, _hi, _def)
        rng = np.random.default_rng(int(rng_seed))
        mat = _subsample_square(A, max_preview, rng)
        vmax = float(np.percentile(np.abs(mat[mat != 0]), 99)) if np.any(mat != 0) else 1.0
        vmax = max(vmax, 1e-6)

        fig1, ax1 = plt.subplots(figsize=(6.2, 5.2))
        im = ax1.imshow(mat, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax1.set_title("Subsampled recurrent adjacency (signed weights)")
        ax1.set_xlabel("neuron index (subsample)")
        ax1.set_ylabel("neuron index (subsample)")
        fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        fig1.tight_layout()
        st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)

        out_deg = np.diff(A.indptr)
        At = A.transpose().tocsr()
        in_deg = np.diff(At.indptr)
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 3.8))
        ax2.hist(out_deg, bins=min(60, max(5, n // 10)), color="#534AB7", alpha=0.85)
        ax2.set_title("Out-degree distribution")
        ax2.set_xlabel("edges per neuron")
        ax2.set_ylabel("count")
        ax3.hist(in_deg, bins=min(60, max(5, n // 10)), color="#1D9E75", alpha=0.85)
        ax3.set_title("In-degree distribution")
        ax3.set_xlabel("edges per neuron")
        ax3.set_ylabel("count")
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

        st.markdown("#### Wiring diagram (interactive)")
        st.caption(
            "**Which model?** This graph is the **ConnectomeSNN recurrent topology** — the signed adjacency in "
            "`olfactory_connectome.npz` (where synapses exist and excitatory vs inhibitory sign). "
            "**ShuffledSNN** uses a degree-preserving **permutation** of this graph (not shown here). "
            "**SparseMLP** / **DenseMLP** do not use this wiring. "
            "When `olfactory_connectome_positions.npz` is present, nodes are placed using **FlyWire L2 bbox centroids** "
            "(nm) and the plot becomes brain-aligned. Otherwise it uses a **force-directed** layout on a local "
            "subgraph. Drag to rotate (3D), scroll to zoom."
        )
        w1, w2, w3 = st.columns(3)
        with w1:
            wire_center = st.number_input("Seed neuron index", min_value=0, max_value=max(0, n - 1), value=min(n // 2, n - 1))
        with w2:
            wire_max_nodes = st.slider("Max neurons in subgraph", 12, 200, 56)
        with w3:
            wire_max_edges = st.slider("Max edges drawn", 40, 2500, 600)
        w4, w5 = st.columns(2)
        with w4:
            wire_dim = st.radio("Layout", ("3D (rotate like Codex)", "2D"), horizontal=True)
        with w5:
            wire_layout_seed = st.number_input("Layout seed", min_value=0, max_value=99_999, value=2025)
        wire_edge_seed = st.number_input("Edge subsample seed (if truncating)", min_value=0, max_value=99_999, value=7)

        try:
            from flywire_snn.viz.wiring import make_wiring_figure

            wire_rng = np.random.default_rng(int(wire_edge_seed))
            fig_w = make_wiring_figure(
                A,
                center_idx=int(wire_center),
                max_nodes=int(wire_max_nodes),
                max_edges=int(wire_max_edges),
                dim=2 if str(wire_dim).startswith("2") else 3,
                layout_seed=int(wire_layout_seed),
                rng=wire_rng,
                positions_nm=positions_nm,
                prefer_brain_layout=True,
            )
            if fig_w is None:
                st.info("No edges in this subgraph — try a different seed neuron or increase max neurons.")
            else:
                st.plotly_chart(fig_w, use_container_width=True)
        except ImportError as e:
            st.warning(f"Install wiring dependencies: `pip install plotly networkx` ({e})")

    st.divider()
    st.subheader("Model architecture summary (from cache + DoOR shape)")
    dims = _door_dims(data_dir)
    if dims is None:
        st.info(
            f"No `{DOOR_CSV}` found under `{data_dir / 'processed'}`. "
            "Use manual dimensions below (must match your experiment)."
        )
        num_classes = st.number_input("Number of odor classes (rows)", min_value=2, max_value=50_000, value=500)
        input_dim = st.number_input("Receptor / input dim (columns)", min_value=2, max_value=2000, value=50)
    else:
        num_classes, input_dim = dims
        st.success(f"DoOR cache: **{num_classes}** odors × **{input_dim}** receptors (as in the experiment).")

    if not cache.exists():
        return

    cfg = ExperimentConfig()
    A = sp.load_npz(cache).tocsr()
    hidden_dim = int(A.shape[0])
    rho = recurrent_sparsity_ratio(A)
    with st.spinner("Building torch modules for parameter counts (may take a few seconds)…"):
        connectome_snn = MaskedRecurrentLIFSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            adjacency=A,
            steps=cfg.snn_steps,
            alpha=cfg.snn_alpha,
        )
        p_snn = _parameter_count_module(connectome_snn)
        shuffled = ShuffledSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            adjacency=A,
            shuffle_seed=cfg.base_shuffle_seed,
            steps=cfg.snn_steps,
            alpha=cfg.snn_alpha,
        )
        sparse_mlp = SparseMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            sparsity_ratio=rho,
            seed=cfg.base_sparse_seed,
        )
        dense = DenseMLP.matched_to_connectome_snn(input_dim, num_classes, p_snn)
        h_dense = dense.net[0].out_features
        p_dense = _parameter_count_module(dense)

    arch_rows = [
        {
            "Model": "ConnectomeSNN",
            "Hidden neurons": hidden_dim,
            "Recurrent mask nnz": int(A.nnz),
            "Recurrent density": f"{rho:.5f}",
            "Trainable params": p_snn,
            "Notes": "Fixed topology + sign; LIF surrogate training",
        },
        {
            "Model": "ShuffledSNN",
            "Hidden neurons": hidden_dim,
            "Recurrent mask nnz": int(A.nnz),
            "Recurrent density": f"{rho:.5f}",
            "Trainable params": _parameter_count_module(shuffled),
            "Notes": "Degree-preserving shuffle of mask",
        },
        {
            "Model": "SparseMLP",
            "Hidden neurons": hidden_dim,
            "Recurrent mask nnz": "—",
            "Recurrent density": f"{rho:.5f} (layer masks)",
            "Trainable params": _parameter_count_module(sparse_mlp),
            "Notes": "Two sparse ReLU layers at ρ ≈ connectome",
        },
        {
            "Model": "DenseMLP",
            "Hidden neurons": h_dense,
            "Recurrent mask nnz": "—",
            "Recurrent density": "1.0 (dense)",
            "Trainable params": p_dense,
            "Notes": f"Width chosen ≈ match ConnectomeSNN ({p_dense} vs {p_snn})",
        },
    ]
    st.dataframe(arch_rows, width="stretch")

    with st.expander("Architecture sketch (text)"):
        st.code(
            f"""ConnectomeSNN / ShuffledSNN (shared layout, different recurrent mask)
  x ∈ R^{input_dim}
    → Linear(no bias): in → N={hidden_dim}
    → LIF surrogate ({cfg.snn_steps} steps) with masked recurrent W_rec (N×N, topology fixed)
    → Linear: N → C={num_classes} (logits)

SparseMLP
  x ∈ R^{input_dim}
    → Linear → N={hidden_dim} (ReLU), weights × random mask (density ρ≈{rho:.4f})
    → Linear → N={hidden_dim} (ReLU), weights × random mask
    → Linear → C={num_classes}

DenseMLP (param-matched to ConnectomeSNN)
  x ∈ R^{input_dim}
    → Linear → H={h_dense} (ReLU) → Linear → H (ReLU) → Linear → C={num_classes}
""",
            language="text",
        )

    st.caption(
        "Heatmap uses a random **subset** of neurons so large connectomes stay responsive. "
        "Degree plots use the **full** graph. Parameter counts use the same constructors as `experiment.py`."
    )


st.set_page_config(page_title="FlyWire Olfactory SNN GUI", layout="wide")
st.title("FlyWire Olfactory SNN Experiment")

if "exp_proc" not in st.session_state:
    st.session_state.exp_proc = None


@st.fragment(run_every=0.5)
def _live_log_block(result_dir: Path) -> None:
    log_path = result_dir / "run.log"
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        max_lines = st.slider("Log lines (tail)", 100, 5000, 800, step=50, key="log_max_lines")
    with c2:
        hide_noise = st.checkbox(
            "Hide noisy third-party DEBUG",
            value=True,
            key="log_hide_noise",
            help="Filters DEBUG lines from libraries like jsonschema when log level is DEBUG.",
        )
    with c3:
        compact = st.checkbox(
            "Compact format",
            value=True,
            key="log_compact",
            help="Shorter timestamps and logger names so training lines stand out.",
        )

    proc = st.session_state.exp_proc
    if proc is not None:
        rc = proc.poll()
        if rc is None:
            st.caption(":green[**Status:** experiment subprocess running — tailing `run.log`…]")
        else:
            st.caption(f":blue[**Status:** subprocess finished — exit code `{rc}`.]")
    else:
        st.caption("**Status:** no subprocess started from this session (showing file tail if present).")

    if not log_path.exists():
        st.info("No `run.log` yet. Start a run from the sidebar, or check the results directory path.")
        return

    raw = _read_log_tail(log_path)
    text = _format_log(raw, max_lines=max_lines, hide_noise=hide_noise, compact=compact)
    st.text_area(
        "run.log",
        value=text,
        height=560,
        disabled=True,
        label_visibility="collapsed",
    )


with st.sidebar:
    st.header("Run configuration")
    result_dir = Path(st.text_input("Results dir", value="results"))
    data_dir = Path(st.text_input("Data dir", value="data"))

    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=80, step=1)
    batch_size = st.number_input("Batch size", min_value=1, max_value=2048, value=32, step=1)
    seed = st.number_input("Base seed", min_value=0, max_value=1_000_000_000, value=7, step=1)
    max_neurons = st.number_input("Max neurons", min_value=10, max_value=5000, value=800, step=10)

    n_folds = st.number_input("CV folds", min_value=2, max_value=20, value=5, step=1)
    n_seeds = st.number_input("Seeds per fold", min_value=1, max_value=50, value=5, step=1)
    patience = st.number_input("Early stopping patience", min_value=1, max_value=50, value=5, step=1)

    annotation_dataset = st.text_input("FlyWire annotation dataset", value="public")
    materialization = st.text_input("Materialization", value="auto")

    col1, col2 = st.columns(2)
    with col1:
        rebuild_connectome = st.checkbox("Rebuild connectome", value=False)
        refresh_door = st.checkbox("Refresh DoOR cache", value=False)
    with col2:
        require_real = st.checkbox("Require real connectome", value=False)

    log_level = st.selectbox("Log level", options=["INFO", "DEBUG", "WARNING", "ERROR"], index=0)

    models_selected = st.multiselect(
        "Models to train",
        options=list(ALL_MODEL_NAMES),
        default=list(ALL_MODEL_NAMES),
        help="Leave all selected for the full comparison, or pick one or more models for a quicker run.",
    )

    run_btn = st.button("Run experiment now", type="primary")
    stop_btn = st.button("Stop experiment subprocess", help="Sends terminate() to the child process if still running.")


tab_run, tab_viz = st.tabs(["Run & results", "Model visualization"])

with tab_run:
    left, right = st.columns([1, 1])

    with left:
        st.header("Live log")
        st.caption(
            "The experiment writes to `results/run.log` (same as the CLI). This panel refreshes ~twice per second while "
            "the page is open. Use **Compact format** and **Hide noisy third-party DEBUG** to read DEBUG runs more easily."
        )

        if stop_btn:
            if st.session_state.exp_proc is not None:
                p = st.session_state.exp_proc
                if p.poll() is None:
                    p.terminate()
                    st.warning("Sent terminate() to the experiment subprocess.")
                st.session_state.exp_proc = None

        if run_btn:
            if not models_selected:
                st.warning("Select at least one model.")
            else:
                result_dir.mkdir(parents=True, exist_ok=True)
                args: list[str] = [
                    "--epochs",
                    str(int(epochs)),
                    "--batch-size",
                    str(int(batch_size)),
                    "--seed",
                    str(int(seed)),
                    "--max-neurons",
                    str(int(max_neurons)),
                    "--data-dir",
                    str(data_dir),
                    "--result-dir",
                    str(result_dir),
                    "--annotation-dataset",
                    str(annotation_dataset),
                    "--materialization",
                    str(materialization),
                    "--n-folds",
                    str(int(n_folds)),
                    "--n-seeds",
                    str(int(n_seeds)),
                    "--early-stopping-patience",
                    str(int(patience)),
                    "--log-level",
                    str(log_level),
                    "--models",
                    *models_selected,
                ]
                if rebuild_connectome:
                    args.append("--rebuild-connectome")
                if require_real:
                    args.append("--require-real-connectome")
                if refresh_door:
                    args.append("--refresh-door-cache")
                old = st.session_state.exp_proc
                if old is not None and old.poll() is None:
                    old.terminate()
                    st.info("Previous subprocess was still running — it was terminated before starting a new run.")

                st.session_state.exp_proc = _start_subprocess(args)
                st.success(
                    f"Started subprocess (PID `{st.session_state.exp_proc.pid}`). "
                    f"Tailing `{result_dir / 'run.log'}` below."
                )

        _live_log_block(result_dir)

    with right:
        st.header("Results browser")
        latest = _latest_comparison_json(result_dir)
        files = sorted(result_dir.glob("comparison-*.json"), key=lambda p: p.name) if result_dir.exists() else []
        options = [str(p) for p in files]
        default_idx = options.index(str(latest)) if latest and str(latest) in options else (len(options) - 1 if options else 0)
        selected = st.selectbox("Select a comparison JSON", options=options, index=default_idx if options else None)

        if selected:
            p = Path(selected)
            try:
                payload = _load_json(p)
                st.caption(f"Loaded: `{p}`")
                _render_summary(payload)
            except Exception as e:
                st.error(f"Failed to load JSON: {e}")
        else:
            st.info("No dated `comparison-*.json` files found yet.")

with tab_viz:
    st.header("Model visualization")
    st.caption(
        "Inspect the **cached FlyWire recurrent graph** and a quick **architecture / parameter** comparison "
        f"(uses `{data_dir / 'processed' / CONNECTOME_NPZ}` and `.../{DOOR_CSV}` when present)."
    )
    _render_model_visualization(data_dir)
