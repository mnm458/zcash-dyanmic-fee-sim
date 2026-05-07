"""CLI runner for the fee simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_scenario
from .scenarios import run_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zfeesim",
        description="Zcash Dynamic Fee Simulator",
    )
    sub = parser.add_subparsers(dest="command")

    # Default: --scenario (backward compat)
    parser.add_argument("--scenario", help="Path to scenario YAML (shortcut for 'run')")
    parser.add_argument("--out-dir", default="results", help="Output directory")

    # sweep subcommand
    p_sweep = sub.add_parser("sweep", help="Run a parameter sweep")
    p_sweep.add_argument("--base", required=True, help="Base scenario YAML")
    p_sweep.add_argument("--param", required=True, help="Dot-path parameter to sweep")
    p_sweep.add_argument("--values", required=True, help="Comma-separated values")
    p_sweep.add_argument("--out-dir", default="results/sweeps")

    # compare subcommand
    p_cmp = sub.add_parser("compare", help="Run a head-to-head comparison")
    p_cmp.add_argument("--configs", nargs="+", required=True, help="label=path pairs")
    p_cmp.add_argument("--out-dir", default="results/comparisons")

    # validate subcommand
    p_val = sub.add_parser("validate", help="Run full validation report")
    p_val.add_argument("--out-dir", default="results/validation")

    # adversarial subcommand
    p_adv = sub.add_parser("adversarial", help="Run adversarial cost optimization")
    p_adv.add_argument("--out-dir", default="results/adversarial")

    # audit subcommand
    p_audit = sub.add_parser("audit", help="Full audit with cost traces, robustness, and summary")
    p_audit.add_argument("--out-dir", default="results/audit")

    args = parser.parse_args()

    if args.scenario:
        _run_single(args.scenario, args.out_dir)
    elif args.command == "sweep":
        _run_sweep(args)
    elif args.command == "compare":
        _run_compare(args)
    elif args.command == "validate":
        _run_validate(args)
    elif args.command == "adversarial":
        _run_adversarial(args)
    elif args.command == "audit":
        _run_audit(args)
    else:
        parser.print_help()


def _run_single(scenario_path: str, out_dir: str) -> None:
    cfg = load_scenario(scenario_path)
    name = cfg.get("name", Path(scenario_path).stem)

    print(f"Running scenario: {name}")
    metrics, chain = run_scenario(cfg)

    out = Path(out_dir)
    csv_override = cfg.get("metrics", {}).get("output_csv")
    csv_path = Path(csv_override) if csv_override else out / f"{name}.csv"
    json_path = out / f"{name}_summary.json"

    metrics.write_csv(csv_path)
    metrics.write_summary_json(json_path)

    summary = metrics.summary()
    print(f"\nscenario: {name}")
    for key in [
        "effective_attacker_cost", "effective_attacker_cost_zec",
        "attacker_nominal_fee_paid", "miner_self_nominal_fee_paid", "miner_recovered_fee",
        "honest_total_fee", "honest_total_fee_zec",
        "honest_overpayment_vs_fixed_zip317", "honest_overpayment_zec",
        "baseline_overpayment", "incremental_overpayment", "incremental_overpayment_zec",
        "harm_ratio", "median_confirmation_delay", "p95_confirmation_delay",
        "expired_honest_transactions", "expired_attacker_transactions",
        "raw_oracle_fee_final", "public_fee_bucket_final",
        "fee_bucket_jumps", "fee_volatility",
        "fast_lane_open_blocks", "fast_lane_flaps",
        "synthetic_displacement_ratio_avg",
    ]:
        print(f"  {key}: {summary.get(key, 'N/A')}")

    print(f"\nOutputs: {csv_path}, {json_path}")


def _run_sweep(args) -> None:
    from .sweep import run_sweep

    base = load_scenario(args.base)
    raw_values = args.values.split(",")
    values = []
    for v in raw_values:
        v = v.strip()
        try:
            values.append(int(v))
        except ValueError:
            try:
                values.append(float(v))
            except ValueError:
                values.append(v)

    print(f"Sweep: {args.param} = {values}")
    results = run_sweep(base, args.param, values, args.out_dir)
    print(f"\n{'value':>12s} | {'harm_ratio':>12s} | {'fee_jumps':>10s} | {'fl_flaps':>10s} | {'med_delay':>10s} | {'cost_zec':>12s}")
    print("-" * 80)
    for r in results:
        print(f"{r['_sweep_value']:>12} | {r['harm_ratio']:>12.4f} | {r['fee_bucket_jumps']:>10d} | {r['fast_lane_flaps']:>10d} | {r['median_confirmation_delay']:>10.2f} | {r['effective_attacker_cost_zec']:>12.6f}")


def _run_compare(args) -> None:
    from .sweep import run_comparison, format_comparison_report

    configs = {}
    for pair in args.configs:
        label, path = pair.split("=", 1)
        configs[label] = load_scenario(path)

    summaries = run_comparison(configs, args.out_dir)
    print(format_comparison_report(summaries, "Comparison"))


def _run_validate(args) -> None:
    from .validation import run_full_validation
    run_full_validation(args.out_dir)


def _run_adversarial(args) -> None:
    from .adversarial import run_adversarial_optimization, format_attack_report
    reports = run_adversarial_optimization(args.out_dir)
    print()
    for r in reports:
        print(format_attack_report(r))


def _run_audit(args) -> None:
    from .audit import run_audit
    run_audit(args.out_dir)


if __name__ == "__main__":
    main()
