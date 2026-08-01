"""
Export the FlyWire olfactory SNN to Hugging Face Hub.

Uploads three resources:
  1. Model repo   — best ConnectomeSNN checkpoint + connectome topology + model code
  2. Dataset repo  — processed DoOR odor × receptor matrix
  3. Space repo    — Streamlit demo app

Usage:
    # Upload everything (will prompt for HF token if not logged in):
    python export_to_hf.py --hf-user YOUR_USERNAME

    # Upload only the model:
    python export_to_hf.py --hf-user YOUR_USERNAME --only model

    # Upload only the dataset:
    python export_to_hf.py --hf-user YOUR_USERNAME --only dataset

    # Dry-run (stage files locally without pushing):
    python export_to_hf.py --hf-user YOUR_USERNAME --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_best_checkpoint(
    checkpoint_dir: Path,
    model_name: str = "ConnectomeSNN",
    dataset: str = "DoOR",
) -> Optional[Path]:
    """Find the checkpoint with the highest test_acc for the given model."""
    candidates = sorted(checkpoint_dir.glob(f"{dataset}_{model_name}_*.pt"))
    if not candidates:
        return None
    best_path, best_acc = None, -1.0
    for p in candidates:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        acc = ckpt.get("test_acc", -1.0)
        if acc > best_acc:
            best_acc = acc
            best_path = p
    return best_path


def _load_model_card(hf_user: str) -> str:
    template = ROOT / "hf_model_card.md"
    text = template.read_text(encoding="utf-8")
    return text.replace("{{HF_USER}}", hf_user)


def _load_dataset_card(hf_user: str) -> str:
    template = ROOT / "hf_dataset_card.md"
    text = template.read_text(encoding="utf-8")
    return text.replace("{{HF_USER}}", hf_user)


# ── Model export ─────────────────────────────────────────────────────────────

def export_model(
    hf_user: str,
    checkpoint_dir: Path,
    connectome_npz: Path,
    connectome_meta: Path,
    dry_run: bool = False,
) -> Path:
    """Stage (and optionally push) the model repo."""
    from safetensors.torch import save_file as save_safetensors

    staging = ROOT / "_hf_staging" / "model"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Find best ConnectomeSNN checkpoint
    best_ckpt_path = _find_best_checkpoint(checkpoint_dir)
    if best_ckpt_path is None:
        print(
            "ERROR: No ConnectomeSNN checkpoints found. "
            "Run the experiment first: python run_experiment.py"
        )
        sys.exit(1)

    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    print(f"Best checkpoint: {best_ckpt_path.name} (test_acc={ckpt.get('test_acc', '?')})")

    # Save weights as safetensors
    save_safetensors(state_dict, str(staging / "model.safetensors"))

    # Save architecture config
    config = {
        "model_type": "MaskedRecurrentLIFSNN",
        "input_dim": ckpt.get("feature_dim"),
        "hidden_dim": ckpt.get("hidden_dim"),
        "num_classes": ckpt.get("num_classes"),
        "snn_steps": ckpt.get("snn_steps", 20),
        "snn_alpha": ckpt.get("snn_alpha", 100.0),
        "test_acc": ckpt.get("test_acc"),
        "best_val_acc": ckpt.get("best_val_acc"),
        "stopped_epoch": ckpt.get("stopped_epoch"),
        "seed": ckpt.get("seed"),
        "fold": ckpt.get("fold"),
    }
    (staging / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Copy connectome topology
    if connectome_npz.exists():
        shutil.copy2(connectome_npz, staging / "connectome_mask.npz")
    if connectome_meta.exists():
        shutil.copy2(connectome_meta, staging / "connectome_meta.json")

    # Copy standalone model file
    snn_src = ROOT / "src" / "flywire_snn" / "models" / "snn.py"
    shutil.copy2(snn_src, staging / "modeling_snn.py")

    # Model card
    readme_text = _load_model_card(hf_user)
    (staging / "README.md").write_text(readme_text, encoding="utf-8")

    print(f"Model staged at: {staging}")

    if not dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = f"{hf_user}/flywire-olfactory-snn"
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload FlyWire olfactory SNN model",
        )
        print(f"Pushed model to: https://huggingface.co/{repo_id}")

    return staging


# ── Dataset export ───────────────────────────────────────────────────────────

def export_dataset(
    hf_user: str,
    data_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Stage (and optionally push) the dataset repo."""
    staging = ROOT / "_hf_staging" / "dataset"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    door_csv = data_dir / "processed" / "door_or_merged.csv"
    if not door_csv.exists():
        print(
            "ERROR: DoOR processed CSV not found at "
            f"{door_csv}. Run the experiment first to build the cache."
        )
        sys.exit(1)

    shutil.copy2(door_csv, staging / "door_or_merged.csv")

    # Dataset card
    readme_text = _load_dataset_card(hf_user)
    (staging / "README.md").write_text(readme_text, encoding="utf-8")

    print(f"Dataset staged at: {staging}")

    if not dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = f"{hf_user}/door-olfactory-responses"
        api.create_repo(repo_id, exist_ok=True, repo_type="dataset")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload DoOR olfactory receptor response dataset",
        )
        print(f"Pushed dataset to: https://huggingface.co/datasets/{repo_id}")

    return staging


# ── Space export ─────────────────────────────────────────────────────────────

def export_space(
    hf_user: str,
    data_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Stage (and optionally push) the Streamlit Space."""
    staging = ROOT / "_hf_staging" / "space"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Copy the Streamlit app
    shutil.copy2(ROOT / "gui_app.py", staging / "app.py")

    # Copy source tree
    src_dest = staging / "src"
    shutil.copytree(ROOT / "src", src_dest, dirs_exist_ok=True)

    # Copy data caches if they exist (so Space can visualize without API)
    data_dest = staging / "data" / "processed"
    data_dest.mkdir(parents=True)
    for fname in ["door_or_merged.csv", "olfactory_connectome.npz",
                   "olfactory_connectome.meta.json",
                   "olfactory_connectome_positions.npz"]:
        src_file = data_dir / "processed" / fname
        if src_file.exists():
            shutil.copy2(src_file, data_dest / fname)

    # Space-specific requirements
    space_reqs = [
        "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "seaborn",
        "torch", "norse", "streamlit", "plotly", "networkx",
    ]
    (staging / "requirements.txt").write_text(
        "\n".join(space_reqs) + "\n", encoding="utf-8"
    )

    # HF Space metadata file
    space_readme = f"""---
title: FlyWire Olfactory SNN Explorer
emoji: 🪰
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.45.0"
app_file: app.py
pinned: false
license: mit
models:
  - {hf_user}/flywire-olfactory-snn
datasets:
  - {hf_user}/door-olfactory-responses
---

# FlyWire Olfactory SNN Explorer

Interactive visualization of the connectome-constrained spiking neural network
for olfactory classification in *Drosophila melanogaster*.

- Inspect the FlyWire recurrent connectivity graph
- View model architecture and parameter counts
- Browse experiment results and learning curves
"""
    (staging / "README.md").write_text(space_readme, encoding="utf-8")

    print(f"Space staged at: {staging}")

    if not dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = f"{hf_user}/flywire-olfactory-snn-demo"
        api.create_repo(repo_id, exist_ok=True, repo_type="space",
                        space_sdk="streamlit")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Upload FlyWire olfactory SNN Streamlit demo",
        )
        print(f"Pushed Space to: https://huggingface.co/spaces/{repo_id}")

    return staging


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Export FlyWire olfactory SNN to Hugging Face Hub")
    p.add_argument("--hf-user", required=True, help="Your Hugging Face username")
    p.add_argument(
        "--only",
        choices=["model", "dataset", "space"],
        default=None,
        help="Upload only one resource (default: all three)",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--result-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage files locally in _hf_staging/ without pushing to HF",
    )
    args = p.parse_args()

    targets = [args.only] if args.only else ["model", "dataset", "space"]
    connectome_npz = args.data_dir / "processed" / "olfactory_connectome.npz"
    connectome_meta = args.data_dir / "processed" / "olfactory_connectome.meta.json"
    checkpoint_dir = args.result_dir / "checkpoints"

    if "model" in targets:
        print("\n── Exporting model ──")
        export_model(args.hf_user, checkpoint_dir, connectome_npz,
                     connectome_meta, dry_run=args.dry_run)

    if "dataset" in targets:
        print("\n── Exporting dataset ──")
        export_dataset(args.hf_user, args.data_dir, dry_run=args.dry_run)

    if "space" in targets:
        print("\n── Exporting Space ──")
        export_space(args.hf_user, args.data_dir, dry_run=args.dry_run)

    print("\nDone.")
    if args.dry_run:
        print("Dry run — files staged in _hf_staging/. Re-run without --dry-run to push.")


if __name__ == "__main__":
    main()
