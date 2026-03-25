from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flywire_snn.config import ALL_MODEL_NAMES

RUNNER = ROOT / "run_experiment.py"

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
    # Logs go to run.log via logging; avoid PIPE buffers filling up.
    return subprocess.Popen(
        [sys.executable, str(RUNNER), *args],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
        skip_hallem = st.checkbox("Skip Hallem secondary", value=False)

    log_level = st.selectbox("Log level", options=["INFO", "DEBUG", "WARNING", "ERROR"], index=0)

    models_selected = st.multiselect(
        "Models to train",
        options=list(ALL_MODEL_NAMES),
        default=list(ALL_MODEL_NAMES),
        help="Leave all selected for the full comparison, or pick one or more models for a quicker run.",
    )

    run_btn = st.button("Run experiment now", type="primary")
    stop_btn = st.button("Stop experiment subprocess", help="Sends terminate() to the child process if still running.")


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
            if skip_hallem:
                args.append("--skip-hallem-secondary")

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
