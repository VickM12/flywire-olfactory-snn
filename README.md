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

- **Primary (default):** **DoOR** — merged receptor matrix from [ropensci/DoOR.data](https://github.com/ropensci/DoOR.data) (`Or*.csv` files), cached under `data/processed/door_or_merged.csv`.
- **Secondary (optional):** **Hallem–Carlson** — place `data/raw/hallem_carlson_2006.csv` if you have it; otherwise a **synthetic 110×24** fallback is generated for smoke tests.

## Experiment protocol

- **Cross-validation:** `K`-fold splits over **odor identities** (`n_folds` default 5). For each fold, odors are divided into an outer **train** pool and outer **test** pool.
- **Train / validation / test (reported `test_acc`):** Built from **outer train odors only** — same class labels for train, val, and test, with **different Gaussian noise trials** per split. Validation is used for **early stopping** (patience on `val_acc`) and **epochs-to-80%** (first epoch where `val_acc ≥ 0.8`). Weights from the **best validation epoch** are restored before final metrics.
- **Held-out generalization (`heldout_acc`):** Evaluated on **outer test odors** (identities not seen during training). Accuracy is **top-1 among held-out classes only** (not over the full label space). Logged per run in `results/run.log`; **not** printed in the ASCII summary table.

**SNN evaluation:** For models with `"SNN"` in the name, validation and test use **Monte Carlo averaging** (5 forward passes) over stochastic spike sampling so metrics are stable.

**Scale:** Default `n_folds=5` and `n_seeds=5` ⇒ **25 runs per model per dataset** (5×5), **100 trainings per dataset** (4 models). With DoOR + Hallem secondary enabled, that doubles.

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

This includes PyTorch, Norse, scientific stack, and **FlyWire/CAVE clients** (`fafbseg`, `caveclient`, `python-dotenv`). If you only need synthetic data and no API, you can install a subset manually — the repo expects the full `requirements.txt` for real connectome pulls.

4. **FlyWire token** (for real connectivity): set `FLYWIRE_TOKEN` or `CAVE_TOKEN` in the environment or copy `.env.example` → `.env` (see below).

5. Run (DoOR primary, Hallem secondary unless skipped):

```powershell
python run_experiment.py --epochs 80 --max-neurons 800
```

## Outputs

| Artifact | Contents |
|----------|----------|
| `results/comparison-YYYY-MM-DD.json` | Full config snapshot (includes `comparison_json_path`), connectome metadata (including **edge count**), dataset provenance, **aggregated mean ± std** per model, and **`per_run`** rows for every fold/seed/model. Filename uses the **local calendar date** when the run finishes (same-day reruns overwrite). |
| `results/run.log` | Per-epoch training lines and final **`test_acc`**, **`heldout_acc`**, spike sparsity, `stopped_epoch`, `best_val_acc`. |
| Console | ASCII summary tables (DoOR and, if enabled, Hallem) — **test accuracy**, epochs to 80% (validation), spike sparsity (SNNs), parameter counts. |

**Summary JSON metrics** (per model, per dataset): `test_acc`, `epochs_to_80` (from validation), `stopped_epoch`, `spike_sparsity` (SNNs only; NaN for MLPs), `params`.

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 80 | Max epochs per run (early stopping may stop sooner). |
| `--batch-size` | 32 | Minibatch size. |
| `--seed` | 7 | Base seed; each of `n_seeds` runs uses a derived seed. |
| `--max-neurons` | 800 | Cap on olfactory subgraph size for FlyWire. |
| `--data-dir` | `data` | Data and caches. |
| `--result-dir` | `results` | Logs and JSON. |
| `--annotation-dataset` | `public` | FlyWire annotation dataset. |
| `--materialization` | `auto` | FlyWire materialization version. |
| `--n-folds` | 5 | CV folds over odor identities. |
| `--n-seeds` | 5 | Random seeds per fold. |
| `--early-stopping-patience` | 5 | Early stopping patience on `val_acc`. |
| `--rebuild-connectome` | off | Ignore cache and rebuild connectome. |
| `--require-real-connectome` | off | Fail if a real FlyWire graph cannot be loaded. |
| `--skip-hallem-secondary` | off | Run **DoOR only** (skip Hallem–Carlson). |
| `--refresh-door-cache` | off | Re-download/merge DoOR CSVs. |
| `--log-level` | `INFO` | Logging verbosity. |

## FlyWire / CAVE API token

Connectivity and annotation queries use a token from [global.daf-apis.com](https://global.daf-apis.com). With `python-dotenv` installed, `run_experiment.py` loads `.env` and registers the secret via `fafbseg` when `FLYWIRE_TOKEN` or `CAVE_TOKEN` is set. **Do not commit `.env`** — it is gitignored.

## Connectome cache

The subgraph loader and cache live under `src/flywire_snn/connectome/`; the processed graph is stored at `data/processed/olfactory_connectome.npz` with metadata. The experiment logs **directed edge count** (nonzeros in the sparse matrix, or `edges` / `edges_kept` from metadata when present).

## Project layout

```
src/flywire_snn/
  config.py          # Experiment hyperparameters
  experiment.py      # CV loops, models, aggregation, JSON
  trainers.py        # Training, early stopping, evaluation
  data/              # DoOR, Hallem, splits
  connectome/        # FlyWire load/cache, auth, degree shuffle
  models/            # SNN, ShuffledSNN, SparseMLP, DenseMLP
run_experiment.py    # CLI entrypoint (adds `src/` to `sys.path`)
```

Core **ConnectomeSNN** implementation: `src/flywire_snn/models/snn.py`.

## Notes

- Tuned for **CPU** training; adjust `--epochs` / `--batch-size` / `--n-folds` / `--n-seeds` for shorter dry runs.
- First DoOR build downloads many CSVs from GitHub (one-time; then cached locally).
