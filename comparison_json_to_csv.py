"""
Export comparison result JSON (e.g. results/comparison-03202026.json) to CSV.

By default writes one table from the ``per_run`` array (one row per fold/seed/model).
Use --all to also emit summary aggregates and config as separate CSVs next to the main file.

Usage:
    python comparison_json_to_csv.py results/comparison-03202026.json
    python comparison_json_to_csv.py results/comparison-03202026.json -o results/my_runs.csv
    python comparison_json_to_csv.py results/comparison-03202026.json --all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _flatten_config(prefix: str, obj: Any, out: list[tuple[str, str]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten_config(key, v, out)
    else:
        out.append((prefix, "" if obj is None else json.dumps(obj) if isinstance(obj, (dict, list)) else str(obj)))


def write_key_value_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)


def write_per_run_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit("JSON has no 'per_run' array or it is empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, Any]] = []
    for dataset, models in summary.items():
        if not isinstance(models, dict):
            continue
        for model, metrics in models.items():
            if not isinstance(metrics, dict):
                continue
            for name, val in metrics.items():
                if isinstance(val, dict) and "mean" in val and "std" in val:
                    out_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "metric": name,
                            "mean": val["mean"],
                            "std": val["std"],
                        }
                    )
                else:
                    out_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "metric": name,
                            "mean": val,
                            "std": "",
                        }
                    )
    fieldnames = ["dataset", "model", "metric", "mean", "std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert comparison JSON to CSV.")
    p.add_argument("json_path", type=Path, help="Path to comparison *.json")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output CSV for per_run (default: <json_stem>_per_run.csv next to JSON)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Also write <stem>_summary.csv and <stem>_config.csv",
    )
    args = p.parse_args()

    json_path: Path = args.json_path
    if not json_path.is_file():
        raise SystemExit(f"File not found: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    stem = json_path.with_suffix("")

    per_run_out = args.out if args.out is not None else Path(f"{stem}_per_run.csv")
    write_per_run_csv(per_run_out, data.get("per_run") or [])

    if args.all:
        summary_path = Path(f"{stem}_summary.csv")
        if "summary" in data and isinstance(data["summary"], dict):
            write_summary_csv(summary_path, data["summary"])
        else:
            summary_path.write_text("", encoding="utf-8")

        config_rows: list[tuple[str, str]] = []
        if "config" in data and isinstance(data["config"], dict):
            _flatten_config("", data["config"], config_rows)
        write_key_value_csv(Path(f"{stem}_config.csv"), config_rows)

    print(f"Wrote {per_run_out.resolve()}")
    if args.all:
        print(f"Wrote {Path(f'{stem}_summary.csv').resolve()}")
        print(f"Wrote {Path(f'{stem}_config.csv').resolve()}")


if __name__ == "__main__":
    main()
