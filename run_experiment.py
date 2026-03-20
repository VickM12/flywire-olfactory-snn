import argparse
import json
from pathlib import Path

from flywire_snn.config import ExperimentConfig
from flywire_snn.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FlyWire connectome SNN vs MLP experiment")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-neurons", type=int, default=800)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--result-dir", type=Path, default=Path("results"))
    p.add_argument("--annotation-dataset", type=str, default="public")
    p.add_argument("--materialization", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_olfactory_neurons=args.max_neurons,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        annotation_dataset=args.annotation_dataset,
        materialization=args.materialization,
    )
    result = run_experiment(cfg)
    print(json.dumps(result["models"], indent=2))
    print(f"Saved results to: {cfg.result_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()

