# FlyWire Olfactory SNN Experiment

Compares a **FlyWire connectome–constrained recurrent LIF SNN** (`MaskedRecurrentLIFSNN`, logged as **ConnectomeSNN**) against publication-oriented baselines on **odor identity classification** (receptor response vectors → class labels). Training uses **Norse** LIF neurons and **surrogate gradients** for the SNNs; baselines are standard ReLU MLPs with different masks / width.

## Models (one `run_experiment.py` invocation)

| Model | Description |
|-------|-------------|
| **ConnectomeSNN** | Recurrent topology fixed to the FlyWire (or cached) adjacency; trainable signed weights, LIF dynamics. |
| **ShuffledSNN** | Same architecture and training; recurrent mask is **degree-preserving shuffled** (isolates biological topology vs. sparsity/degree). |
| **SparseMLP** | Two hidden layers, ReLU; fixed **random** sparse masks at the **same density** as the connectome recurrent weights. |
| **DenseMLP** | Fully connected ReLU MLP; hidden width chosen to **match ConnectomeSNN parameter count** (approximately). |

## Data

**DoOR** — merged receptor matrix from [ropensci/DoOR.data](https://github.com/ropensci/DoOR.data) (`Or*.csv` files), cached under `data/processed/door_or_merged.csv`. Licensed CC BY-SA 4.0.

## Experiment protocol

- **Cross-validation:** `K`-fold splits over **odor identities** (`n_folds` default 5). For each fold, odors are divided into an outer **train** pool and outer **test** pool.
- **Train / validation / test (reported `test_acc`):** Built from **outer train odors only** — same class labels for train, val, and test, with **different Gaussian noise trials** per split. Validation is used for **early stopping** (patience on `val_acc`) and **epochs-to-80%** (first epoch where `val_acc ≥ 0.8`). Weights from the **best validation epoch** are restored before final metrics.
- **Held-out generalization (`heldout_acc`):** Evaluated on **outer test odors** (identities not seen during training). Accuracy is **top-1 among held-out classes only** (not over the full label space). Logged per run in `results/run.log`; **not** printed in the ASCII summary table.

**SNN evaluation:** For models with `"SNN"` in the name, validation and test use **Monte Carlo averaging** (5 forward passes) over stochastic spike sampling so metrics are stable.

**Scale:** Default `n_folds=5` and `n_seeds=5` → **25 runs per model** (5×5), **100 trainings total** (4 models).

## Quick start

1. Python **3.10+** recommended.

2. Create and activate a venv (example for PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. **FlyWire token** (for real connectivity): set `FLYWIRE_TOKEN` or `CAVE_TOKEN` in the environment or copy `.env.example` → `.env`.

5. Run:

```powershell
python run_experiment.py --epochs 80 --max-neurons 800
```

## GUI

```powershell
streamlit run gui_app.py
```

The sidebar includes a **Models to train** multiselect. The **Live log** panel tails `results/run.log` about twice per second while the app is open. The **Model visualization** tab plots the cached connectome (subsampled weight heatmap, in/out degree histograms), an **interactive wiring diagram** (Plotly), and a parameter-count / architecture summary.

## Hugging Face upload

After running the experiment, export the model, dataset, and a Streamlit Space demo to Hugging Face Hub:

```powershell
# Dry-run (stages files locally in _hf_staging/ without pushing):
python export_to_hf.py --hf-user YOUR_USERNAME --dry-run

# Push everything:
python export_to_hf.py --hf-user YOUR_USERNAME

# Push only the model, dataset, or Space:
python export_to_hf.py --hf-user YOUR_USERNAME --only model
python export_to_hf.py --hf-user YOUR_USERNAME --only dataset
python export_to_hf.py --hf-user YOUR_USERNAME --only space
```

This creates three HF repos:

| Resource | Repo ID | Contents |
|----------|---------|----------|
| **Model** | `YOUR_USERNAME/flywire-olfactory-snn` | Best ConnectomeSNN weights (safetensors), connectome topology, config, model card |
| **Dataset** | `YOUR_USERNAME/door-olfactory-responses` | Processed DoOR odor × receptor CSV, dataset card (CC BY-SA 4.0) |
| **Space** | `YOUR_USERNAME/flywire-olfactory-snn-demo` | Streamlit app for interactive connectome visualization |

You must be logged in to HF (`huggingface-cli login`) before pushing.

## Outputs

| Artifact | Contents |
|----------|----------|
| `results/comparison-YYYY-MM-DD.json` | Full config, connectome metadata, aggregated mean ± std per model, and `per_run` rows for every fold/seed/model. |
| `results/run.log` | Per-epoch training lines and final `test_acc`, `heldout_acc`, spike sparsity, `stopped_epoch`, `best_val_acc`. |
| `results/checkpoints/` | Per-run model checkpoints (`.pt` files with state dict + metadata). |
| Console | ASCII summary table — test accuracy, epochs to 80%, spike sparsity, parameter counts. |

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 80 | Max epochs per run (early stopping may stop sooner). |
| `--batch-size` | 32 | Minibatch size. |
| `--seed` | 7 | Base seed; each of `n_seeds` runs uses a derived seed. |
| `--max-neurons` | 800 | Cap on olfactory subgraph size for FlyWire. |
| `--data-dir` | `data` | Data and caches. |
| `--result-dir` | `results` | Logs, JSON, and checkpoints. |
| `--annotation-dataset` | `public` | FlyWire annotation dataset. |
| `--materialization` | `auto` | FlyWire materialization version. |
| `--n-folds` | 5 | CV folds over odor identities. |
| `--n-seeds` | 5 | Random seeds per fold. |
| `--early-stopping-patience` | 5 | Early stopping patience on `val_acc`. |
| `--rebuild-connectome` | off | Ignore cache and rebuild connectome. |
| `--require-real-connectome` | off | Fail if a real FlyWire graph cannot be loaded. |
| `--refresh-door-cache` | off | Re-download/merge DoOR CSVs. |
| `--no-fetch-positions` | off | Skip FlyWire L2 centroid fetch (faster rebuild; wiring falls back to force layout). |
| `--log-level` | `INFO` | Logging verbosity. |
| `--models` … | all four | Train only listed models: `ConnectomeSNN`, `ShuffledSNN`, `SparseMLP`, `DenseMLP`. |

## Project layout

```
src/flywire_snn/
  config.py          # Experiment hyperparameters
  experiment.py      # CV loops, models, aggregation, checkpoint saving, JSON
  trainers.py        # Training, early stopping, evaluation
  data/              # DoOR loader, CV splits
  connectome/        # FlyWire load/cache, auth, degree shuffle
  models/            # SNN, ShuffledSNN, SparseMLP, DenseMLP
  viz/               # Interactive wiring diagrams
run_experiment.py    # CLI entrypoint
gui_app.py           # Streamlit GUI
export_to_hf.py      # Hugging Face Hub export (model + dataset + Space)
hf_model_card.md     # Model card template
hf_dataset_card.md   # Dataset card template
```

Core **ConnectomeSNN** implementation: `src/flywire_snn/models/snn.py`.

## Notes

- Tuned for **CPU** training; adjust `--epochs` / `--batch-size` / `--n-folds` / `--n-seeds` for shorter dry runs.
- First DoOR build downloads many CSVs from GitHub (one-time; then cached locally).
- Model checkpoints are saved to `results/checkpoints/` after each training run.
