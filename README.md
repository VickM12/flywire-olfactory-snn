# FlyWire Olfactory SNN Experiment

This project compares a **FlyWire connectome–constrained recurrent LIF SNN** (`MaskedRecurrentLIFSNN` / **ConnectomeSNN** in logs) against publication-oriented baselines on odor classification.

**Models (single `run_experiment.py` run):**

| Model | Description |
|-------|-------------|
| **ConnectomeSNN** | Connectome-masked recurrent LIF SNN (Norse + surrogate gradients). |
| **ShuffledSNN** | Same architecture and dynamics; recurrent mask is **degree-preserving shuffled** (isolates topology vs sparsity). |
| **SparseMLP** | Two-hidden-layer ReLU MLP; fixed **random** masks at the **same density** as the connectome recurrent matrix. |
| **DenseMLP** | Fully connected ReLU MLP; hidden width chosen to **match ConnectomeSNN parameter count** (approximately). |

**Data:**

- **Primary:** **DoOR** — merged matrix built from [ropensci/DoOR.data](https://github.com/ropensci/DoOR.data) `Or*.csv` files (cached as `data/processed/door_or_merged.csv`).
- **Secondary:** **Hallem–Carlson** — `data/raw/hallem_carlson_2006.csv` (or synthetic 110×24 fallback if missing).

**Protocol:** 5-fold CV × 5 seeds (25 runs per model per dataset), **early stopping** (patience 5 on validation accuracy, restore best weights). Summary = mean ± std; JSON includes every run in `per_run`.

## Quick start

1. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
pip install -r requirements-flywire.txt
```

3. **FlyWire token** (for real connectivity): set `FLYWIRE_TOKEN` or `CAVE_TOKEN` (see `.env.example` and section below).

4. Run (DoOR primary + Hallem secondary by default):

```powershell
python run_experiment.py --epochs 80 --max-neurons 800
```

Outputs:

- `results/comparison.json` — full summary + `per_run` table
- `results/run.log` — training logs
- Printed ASCII summary tables for DoOR and Hallem

### Useful flags

```text
--n-folds 5 --n-seeds 5 --early-stopping-patience 5
--skip-hallem-secondary
--refresh-door-cache
--rebuild-connectome --require-real-connectome
```

### FlyWire / CAVE API token

Connectivity queries need a token from [global.daf-apis.com](https://global.daf-apis.com). `run_experiment.py` loads `.env` (if `python-dotenv` is installed) and calls `fafbseg.flywire.set_chunkedgraph_secret` when `FLYWIRE_TOKEN` or `CAVE_TOKEN` is set.

## Connectome

The FlyWire subgraph loader lives in `src/flywire_snn/connectome/flywire_graph.py` (cached at `data/processed/olfactory_connectome.npz`). Experiment code records **directed edge count** as `connectome.nnz` when metadata uses `edges_kept` instead of `edges`.

## Python compatibility

- Python 3.10+ recommended.
- First DoOR build downloads many CSVs from GitHub (one-time; then uses local cache).

## Notes

- **ConnectomeSNN** implementation: `src/flywire_snn/models/snn.py` (unchanged by baseline additions).
- Training uses **CPU**-friendly defaults; adjust `--batch-size` / `--epochs` as needed.
