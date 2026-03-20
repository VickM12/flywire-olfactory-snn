import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flywire_snn.config import ExperimentConfig
from flywire_snn.experiment import run_experiment


def configure_logging(result_dir: Path, level_name: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(result_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


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
    p.add_argument("--rebuild-connectome", action="store_true")
    p.add_argument("--require-real-connectome", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.result_dir, args.log_level)
    cfg = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_olfactory_neurons=args.max_neurons,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        annotation_dataset=args.annotation_dataset,
        materialization=args.materialization,
        rebuild_connectome=args.rebuild_connectome,
        require_real_connectome=args.require_real_connectome,
    )
    logging.getLogger(__name__).info("Run started")
    result = run_experiment(cfg)
    print(json.dumps(result["models"], indent=2))
    print(f"Saved results to: {cfg.result_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()

