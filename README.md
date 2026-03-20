# FlyWire Olfactory SNN Experiment

This project compares:

- a FlyWire-connectome-constrained recurrent LIF SNN (Norse + surrogate gradients), and
- a parameter-matched unconstrained MLP baseline

on a Drosophila odor classification task inspired by Hallem & Carlson (2006).

## Quick start

1. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Optional (for real FlyWire pulls instead of fallback graph):

```powershell
pip install -r requirements-flywire.txt
```

4. Run:

```powershell
python run_experiment.py --epochs 60 --max-neurons 800
```

Results are written to `results/comparison.json`.
Training logs are written to `results/run.log` and echoed to console.

If you changed graph settings (like `--max-neurons`) and want a fresh FlyWire pull:

```powershell
python run_experiment.py --epochs 60 --max-neurons 800 --rebuild-connectome
```

To fail fast when FlyWire connectivity is unavailable (instead of silently using fallback):

```powershell
python run_experiment.py --epochs 60 --max-neurons 800 --rebuild-connectome --require-real-connectome
```

## Data inputs

- Place Hallem-style data at `data/raw/hallem_carlson_2006.csv`.
- Expected format: first column `odor`, remaining 24 receptor response columns.

If the CSV is missing, the pipeline falls back to a synthetic 110x24 matrix so the code remains runnable.

## Connectome construction

`src/flywire_snn/connectome/flywire_graph.py`:

- queries FlyWire annotations for `ALPN` + `Kenyon_Cell`,
- fetches in-subgraph connectivity using `fafbseg-py`,
- builds a signed sparse matrix from transmitter labels,
- caches as `data/processed/olfactory_connectome.npz`.

If FlyWire access fails (credentials/network/materialization mismatch), a sparse random fallback graph is generated to keep the training flow testable.

## Python compatibility

- Base experiment stack works on Python 3.10+.
- FlyWire deps can be more version-sensitive; if optional FlyWire install fails on your interpreter, run the base stack first and continue with fallback graph mode.

## Notes

- The recurrent SNN topology is fixed by the connectome mask; only magnitudes are trainable.
- Sign is preserved by multiplying trainable magnitudes with a fixed `sign` mask.
- Training is CPU-friendly by default (small batches, moderate time steps).

