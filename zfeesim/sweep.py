"""Parameter sweep runner and comparison report generator."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

from .scenarios import run_scenario


def _set_nested(cfg: dict, dotpath: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated path.

    Examples:
        _set_nested(cfg, "chain.block_action_cap", 500)
        _set_nested(cfg, "attacker.target_fee_multiplier", 20)
    """
    keys = dotpath.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def run_sweep(
    base_cfg: dict[str, Any],
    param_path: str,
    values: list,
    out_dir: str | Path = "results/sweeps",
) -> list[dict]:
    """Run base_cfg once per value, varying param_path each time.

    Returns a list of summary dicts, one per value.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for val in values:
        cfg = copy.deepcopy(base_cfg)
        _set_nested(cfg, param_path, val)
        label = f"{param_path}={val}"
        cfg["name"] = label

        metrics, _ = run_scenario(cfg)
        s = metrics.summary()
        s["_sweep_param"] = param_path
        s["_sweep_value"] = val
        results.append(s)

        metrics.write_summary_json(out_dir / f"{param_path.replace('.', '_')}_{val}_summary.json")

    # Write combined CSV
    if results:
        all_keys = list(results[0].keys())
        # Exclude nested dicts from CSV
        flat_keys = [k for k in all_keys if not isinstance(results[0][k], (dict, list))]
        csv_path = out_dir / f"sweep_{param_path.replace('.', '_')}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k) for k in flat_keys})

    return results


def run_comparison(
    configs: dict[str, dict[str, Any]],
    out_dir: str | Path = "results/comparisons",
) -> dict[str, dict]:
    """Run multiple named configs and produce a comparison summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    for label, cfg in configs.items():
        cfg = copy.deepcopy(cfg)
        cfg["name"] = label
        metrics, _ = run_scenario(cfg)
        s = metrics.summary()
        summaries[label] = s
        metrics.write_csv(out_dir / f"{label}.csv")
        metrics.write_summary_json(out_dir / f"{label}_summary.json")

    # Write comparison table
    if summaries:
        labels = list(summaries.keys())
        first = summaries[labels[0]]
        flat_keys = [k for k in first if not isinstance(first[k], (dict, list))]
        csv_path = out_dir / "comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"] + labels)
            for k in flat_keys:
                row = [k] + [summaries[lb].get(k, "") for lb in labels]
                writer.writerow(row)

    return summaries


def format_comparison_report(summaries: dict[str, dict], title: str = "") -> str:
    """Produce a human-readable markdown comparison table."""
    labels = list(summaries.keys())
    if not labels:
        return ""

    first = summaries[labels[0]]
    highlight_keys = [
        "effective_attacker_cost", "effective_attacker_cost_zec",
        "honest_overpayment_vs_fixed_zip317", "honest_overpayment_zec",
        "harm_ratio",
        "median_confirmation_delay", "p95_confirmation_delay",
        "expired_honest_transactions",
        "fee_bucket_jumps", "fee_volatility",
        "fast_lane_open_blocks", "fast_lane_flaps",
        "synthetic_displacement_ratio_avg",
    ]
    keys = [k for k in highlight_keys if k in first]

    lines: list[str] = []
    if title:
        lines.append(f"## {title}")
        lines.append("")

    header = "| metric | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join(["---"] * len(labels)) + "|"
    lines.append(header)
    lines.append(sep)
    for k in keys:
        vals = []
        for lb in labels:
            v = summaries[lb].get(k, "")
            vals.append(str(v))
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)
