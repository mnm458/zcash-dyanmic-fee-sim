"""Adversarial optimization: find minimum-cost attack configs that achieve target harm.

For each attack strategy, searches over (actions_per_block, target_fee_multiplier)
to find the cheapest configuration that meets a harm condition, then checks
whether the result survives four defense variants.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .oracle import zats_to_zec
from .scenarios import run_scenario


# -----------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------

@dataclass
class AttackResult:
    """One evaluated attack configuration."""
    actions_per_block: int
    fee_multiplier: float
    effective_cost: int
    effective_cost_zec: float
    honest_overpayment: int
    honest_overpayment_zec: float
    harm_ratio: float
    fee_bucket_jumps: int
    fast_lane_open_blocks: int
    fast_lane_flaps: int
    meets_target: bool
    summary: dict = field(repr=False, default_factory=dict)


@dataclass
class OptResult:
    """Result of optimizing one attack against one controller."""
    attack_name: str
    target_description: str
    best: AttackResult | None
    all_evaluated: list[AttackResult] = field(default_factory=list)


@dataclass
class DefenseProbe:
    """Whether a successful attack survives a specific defense variant."""
    defense_name: str
    controller_cfg: dict
    still_meets_target: bool
    harm_ratio: float
    fee_bucket_jumps: int
    fast_lane_open_blocks: int
    effective_cost_zec: float


@dataclass
class FullAttackReport:
    """Complete report for one attack type: optimized result + defense probes."""
    opt: OptResult
    defenses: list[DefenseProbe] = field(default_factory=list)


# -----------------------------------------------------------------------
# Grid search core
# -----------------------------------------------------------------------

def _base_cfg() -> dict:
    return {
        "num_blocks": 400,
        "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "action_weighted_median",
            "quantization": "power_of_10",
            "base_fee": 5000, "floor_fee": 5000,
            "oracle_include_synthetic": True,
        },
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": True, "actions_per_block": 100, "fee_per_action": 5000},
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 20},
        "attacker": {"enabled": False},
    }


def _evaluate(
    base: dict,
    attacker_cfg: dict,
    controller_override: dict | None = None,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["attacker"] = attacker_cfg
    if controller_override:
        cfg["controller"] = controller_override
    m, _ = run_scenario(cfg)
    return m.summary()


HarmCondition = Callable[[dict], bool]


def grid_search(
    base: dict,
    attacker_type: str,
    actions_values: list[int],
    multiplier_values: list[float],
    harm_condition: HarmCondition,
    extra_attacker_params: dict | None = None,
) -> OptResult:
    """Search over (actions_per_block, fee_multiplier) grid.

    Returns the cheapest config that satisfies harm_condition.
    """
    extra = extra_attacker_params or {}
    results: list[AttackResult] = []

    for apb in actions_values:
        for mult in multiplier_values:
            atk_cfg = {
                "enabled": True,
                "type": attacker_type,
                "start_height": 100,
                "end_height": 200,
                "actions_per_block": apb,
                "target_fee_multiplier": mult,
                **extra,
            }
            s = _evaluate(base, atk_cfg)
            meets = harm_condition(s)
            ar = AttackResult(
                actions_per_block=apb,
                fee_multiplier=mult,
                effective_cost=s["effective_attacker_cost"],
                effective_cost_zec=s["effective_attacker_cost_zec"],
                honest_overpayment=s["honest_overpayment_vs_fixed_zip317"],
                honest_overpayment_zec=s["honest_overpayment_zec"],
                harm_ratio=s["harm_ratio"],
                fee_bucket_jumps=s["fee_bucket_jumps"],
                fast_lane_open_blocks=s["fast_lane_open_blocks"],
                fast_lane_flaps=s["fast_lane_flaps"],
                meets_target=meets,
                summary=s,
            )
            results.append(ar)

    # Pick cheapest that meets condition
    feasible = [r for r in results if r.meets_target and r.effective_cost > 0]
    feasible.sort(key=lambda r: r.effective_cost)
    best = feasible[0] if feasible else None

    return OptResult(
        attack_name=attacker_type,
        target_description="",
        best=best,
        all_evaluated=results,
    )


# -----------------------------------------------------------------------
# Defense probing
# -----------------------------------------------------------------------

DEFENSE_VARIANTS: dict[str, dict] = {
    "action_weighted": {
        "type": "ComparableMedianController",
        "lookback": 50, "reorg_buffer": 5,
        "oracle": "action_weighted_median",
        "quantization": "power_of_10",
        "base_fee": 5000, "floor_fee": 5000,
        "oracle_include_synthetic": True,
    },
    "capped_oracle": {
        "type": "ComparableMedianWithCapController",
        "lookback": 50, "reorg_buffer": 5,
        "base_fee": 5000, "floor_fee": 5000,
        "oracle_include_synthetic": True,
    },
    "hysteresis": {
        "type": "ComparableMedianHysteresisController",
        "lookback": 50, "reorg_buffer": 5,
        "oracle": "action_weighted_median",
        "base_fee": 5000, "floor_fee": 5000,
        "move_up_consecutive": 5, "move_down_consecutive": 20,
        "oracle_include_synthetic": True,
    },
    "aimd": {
        "type": "AIMDBucketController",
        "base_fee": 5000,
        "alpha": 0.25, "beta": 0.90,
        "target_utilization": 0.70, "lower_utilization": 0.40,
        "increase_window": 5, "decrease_window": 20,
        "block_action_cap": 1000,
    },
}


def probe_defenses(
    base: dict,
    best: AttackResult,
    attacker_type: str,
    harm_condition: HarmCondition,
    extra_attacker_params: dict | None = None,
) -> list[DefenseProbe]:
    extra = extra_attacker_params or {}
    atk_cfg = {
        "enabled": True,
        "type": attacker_type,
        "start_height": 100,
        "end_height": 200,
        "actions_per_block": best.actions_per_block,
        "target_fee_multiplier": best.fee_multiplier,
        **extra,
    }
    probes: list[DefenseProbe] = []
    for name, ctrl_cfg in DEFENSE_VARIANTS.items():
        s = _evaluate(base, atk_cfg, controller_override=ctrl_cfg)
        probes.append(DefenseProbe(
            defense_name=name,
            controller_cfg=ctrl_cfg,
            still_meets_target=harm_condition(s),
            harm_ratio=s["harm_ratio"],
            fee_bucket_jumps=s["fee_bucket_jumps"],
            fast_lane_open_blocks=s["fast_lane_open_blocks"],
            effective_cost_zec=s["effective_attacker_cost_zec"],
        ))
    return probes


# -----------------------------------------------------------------------
# Per-attack optimization routines
# -----------------------------------------------------------------------

def optimize_median_poisoning(base: dict | None = None) -> FullAttackReport:
    base = base or _base_cfg()
    target = ">=1 fee_bucket_jump"

    def cond(s: dict) -> bool:
        return s["fee_bucket_jumps"] >= 2  # at least one *attack-induced* jump

    opt = grid_search(
        base, "MedianPoisoningAttacker",
        actions_values=[50, 100, 200, 300, 500],
        multiplier_values=[2, 5, 10, 20],
        harm_condition=cond,
    )
    opt.target_description = target

    defenses = []
    if opt.best:
        defenses = probe_defenses(base, opt.best, "MedianPoisoningAttacker", cond)
    return FullAttackReport(opt=opt, defenses=defenses)


def optimize_burst_spam(base: dict | None = None) -> FullAttackReport:
    base = base or _base_cfg()
    target = "harm_ratio > 1"

    def cond(s: dict) -> bool:
        return s["harm_ratio"] > 1.0

    opt = grid_search(
        base, "BurstSpamAttacker",
        actions_values=[50, 100, 200, 500, 800],
        multiplier_values=[2, 5, 10, 20],
        harm_condition=cond,
    )
    opt.target_description = target

    defenses = []
    if opt.best:
        defenses = probe_defenses(base, opt.best, "BurstSpamAttacker", cond)
    return FullAttackReport(opt=opt, defenses=defenses)


def optimize_fast_lane_griefing(base: dict | None = None, min_open_blocks: int = 20) -> FullAttackReport:
    base = base or _base_cfg()
    # Use BinaryFastLaneController for this attack
    base = copy.deepcopy(base)
    base["controller"] = {
        "type": "BinaryFastLaneController",
        "base_fee": 5000, "fast_lane_multiplier": 10,
        "open_threshold": 0.95, "close_threshold": 0.95,
        "use_hysteresis": False,
        "synthetic_actions_per_block": 100, "block_action_cap": 1000,
    }
    base["chain"]["block_action_cap"] = 200
    target = f"fast_lane_open_blocks >= {min_open_blocks}"

    def cond(s: dict) -> bool:
        return s["fast_lane_open_blocks"] >= min_open_blocks

    opt = grid_search(
        base, "FastLaneFlapAttacker",
        actions_values=[50, 80, 90, 95, 100, 120, 150],
        multiplier_values=[1, 2, 5],
        harm_condition=cond,
        extra_attacker_params={"expiry_blocks": 2},
    )
    opt.target_description = target

    # For defense probing, override the fast-lane controller variants
    defenses = []
    if opt.best:
        fl_defenses: dict[str, dict] = {
            "binary_no_hysteresis": {
                "type": "BinaryFastLaneController", "base_fee": 5000,
                "open_threshold": 0.95, "close_threshold": 0.95,
                "use_hysteresis": False,
                "synthetic_actions_per_block": 100, "block_action_cap": 200,
            },
            "binary_hysteresis": {
                "type": "BinaryFastLaneController", "base_fee": 5000,
                "open_threshold": 0.95, "close_threshold": 0.70,
                "use_hysteresis": True,
                "open_consecutive": 10, "close_consecutive": 30,
                "synthetic_actions_per_block": 100, "block_action_cap": 200,
            },
            "priority_buckets": {
                "type": "PriorityBucketController", "base_fee": 5000,
                "open_threshold": 0.95,
                "synthetic_actions_per_block": 100,
            },
        }
        atk_cfg = {
            "enabled": True, "type": "FastLaneFlapAttacker",
            "start_height": 100, "end_height": 200,
            "actions_per_block": opt.best.actions_per_block,
            "target_fee_multiplier": opt.best.fee_multiplier,
            "expiry_blocks": 2,
        }
        for name, ctrl in fl_defenses.items():
            s = _evaluate(base, atk_cfg, controller_override=ctrl)
            defenses.append(DefenseProbe(
                defense_name=name,
                controller_cfg=ctrl,
                still_meets_target=cond(s),
                harm_ratio=s["harm_ratio"],
                fee_bucket_jumps=s["fee_bucket_jumps"],
                fast_lane_open_blocks=s["fast_lane_open_blocks"],
                effective_cost_zec=s["effective_attacker_cost_zec"],
            ))

    return FullAttackReport(opt=opt, defenses=defenses)


def optimize_bucket_nudging(base: dict | None = None, min_jumps: int = 3) -> FullAttackReport:
    base = base or _base_cfg()
    target = f"fee_bucket_jumps >= {min_jumps}"

    def cond(s: dict) -> bool:
        return s["fee_bucket_jumps"] >= min_jumps

    opt = grid_search(
        base, "BucketBoundaryNudgingAttacker",
        actions_values=[20, 50, 100, 200, 300],
        multiplier_values=[1.2, 1.5, 2.0, 3.0, 5.0],
        harm_condition=cond,
        extra_attacker_params={"nudge_fee_multiplier": 0},  # overridden below
    )
    # The grid_search uses target_fee_multiplier, but BucketBoundaryNudgingAttacker
    # uses nudge_fee_multiplier.  Re-run with proper parameter name.
    results: list[AttackResult] = []
    for apb in [20, 50, 100, 200, 300]:
        for mult in [1.2, 1.5, 2.0, 3.0, 5.0]:
            atk_cfg = {
                "enabled": True,
                "type": "BucketBoundaryNudgingAttacker",
                "start_height": 100, "end_height": 300,
                "nudge_actions": apb,
                "nudge_fee_multiplier": mult,
            }
            s = _evaluate(base, atk_cfg)
            meets = cond(s)
            results.append(AttackResult(
                actions_per_block=apb,
                fee_multiplier=mult,
                effective_cost=s["effective_attacker_cost"],
                effective_cost_zec=s["effective_attacker_cost_zec"],
                honest_overpayment=s["honest_overpayment_vs_fixed_zip317"],
                honest_overpayment_zec=s["honest_overpayment_zec"],
                harm_ratio=s["harm_ratio"],
                fee_bucket_jumps=s["fee_bucket_jumps"],
                fast_lane_open_blocks=s["fast_lane_open_blocks"],
                fast_lane_flaps=s["fast_lane_flaps"],
                meets_target=meets,
                summary=s,
            ))

    feasible = [r for r in results if r.meets_target and r.effective_cost > 0]
    feasible.sort(key=lambda r: r.effective_cost)
    best = feasible[0] if feasible else None

    opt = OptResult(
        attack_name="BucketBoundaryNudgingAttacker",
        target_description=target,
        best=best,
        all_evaluated=results,
    )

    defenses = []
    if best:
        atk_cfg = {
            "enabled": True, "type": "BucketBoundaryNudgingAttacker",
            "start_height": 100, "end_height": 300,
            "nudge_actions": best.actions_per_block,
            "nudge_fee_multiplier": best.fee_multiplier,
        }
        for name, ctrl in DEFENSE_VARIANTS.items():
            s = _evaluate(base, atk_cfg, controller_override=ctrl)
            defenses.append(DefenseProbe(
                defense_name=name,
                controller_cfg=ctrl,
                still_meets_target=cond(s),
                harm_ratio=s["harm_ratio"],
                fee_bucket_jumps=s["fee_bucket_jumps"],
                fast_lane_open_blocks=s["fast_lane_open_blocks"],
                effective_cost_zec=s["effective_attacker_cost_zec"],
            ))

    return FullAttackReport(opt=opt, defenses=defenses)


# -----------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------

def format_attack_report(report: FullAttackReport) -> str:
    lines: list[str] = []
    opt = report.opt
    lines.append(f"### {opt.attack_name}")
    lines.append(f"**Target:** {opt.target_description}")
    lines.append("")

    if opt.best is None:
        lines.append("**Result:** No feasible configuration found in the search grid.")
        lines.append("")
        # Show best near-miss
        near = sorted(opt.all_evaluated, key=lambda r: r.effective_cost if r.effective_cost > 0 else float('inf'))
        if near:
            b = near[0]
            lines.append(f"Closest attempt: actions={b.actions_per_block}, mult={b.fee_multiplier}, "
                         f"cost={b.effective_cost_zec:.4f} ZEC, harm_ratio={b.harm_ratio:.4f}, "
                         f"jumps={b.fee_bucket_jumps}, fl_open={b.fast_lane_open_blocks}")
        lines.append("")
        return "\n".join(lines)

    b = opt.best
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| min effective_attacker_cost | {b.effective_cost:,} zats ({b.effective_cost_zec:.4f} ZEC) |")
    lines.append(f"| honest_overpayment | {b.honest_overpayment:,} zats ({b.honest_overpayment_zec:.4f} ZEC) |")
    lines.append(f"| harm_ratio | {b.harm_ratio:.4f} |")
    lines.append(f"| required actions/block | {b.actions_per_block} |")
    lines.append(f"| required fee multiplier | {b.fee_multiplier}x |")
    lines.append(f"| attack duration | {100} blocks (h100-h200) |")
    lines.append(f"| fee_bucket_jumps | {b.fee_bucket_jumps} |")
    lines.append(f"| fast_lane_open_blocks | {b.fast_lane_open_blocks} |")
    lines.append("")

    if report.defenses:
        lines.append("**Defense survival:**")
        lines.append("")
        lines.append("| defense | survives? | harm_ratio | fee_jumps | fl_open | cost_zec |")
        lines.append("|---|---|---|---|---|---|")
        for d in report.defenses:
            surv = "YES" if d.still_meets_target else "NO"
            lines.append(f"| {d.defense_name} | {surv} | {d.harm_ratio:.4f} | "
                         f"{d.fee_bucket_jumps} | {d.fast_lane_open_blocks} | {d.effective_cost_zec:.4f} |")
        lines.append("")

        survived = sum(1 for d in report.defenses if d.still_meets_target)
        total = len(report.defenses)
        if survived == 0:
            lines.append(f"**All {total} defenses block this attack.**")
        elif survived == total:
            lines.append(f"**Attack survives all {total} defenses — mechanism is fragile.**")
        else:
            lines.append(f"**Attack survives {survived}/{total} defenses.**")
        lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Full adversarial report
# -----------------------------------------------------------------------

def run_adversarial_optimization(out_dir: str | Path = "results/adversarial") -> list[FullAttackReport]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = _base_cfg()
    reports: list[FullAttackReport] = []

    attack_runs = [
        ("Median Poisoning", optimize_median_poisoning),
        ("Burst Spam", optimize_burst_spam),
        ("Fast-Lane Griefing", optimize_fast_lane_griefing),
        ("Bucket Boundary Nudging", optimize_bucket_nudging),
    ]

    md_lines: list[str] = []
    md_lines.append("# Adversarial Optimization Report")
    md_lines.append("")

    all_json: list[dict] = []

    for label, fn in attack_runs:
        print(f"  Optimizing: {label} ...")
        report = fn(base)
        reports.append(report)
        md_lines.append(format_attack_report(report))

        # JSON record
        entry: dict = {
            "attack": report.opt.attack_name,
            "target": report.opt.target_description,
            "feasible": report.opt.best is not None,
        }
        if report.opt.best:
            b = report.opt.best
            entry.update({
                "min_effective_cost_zats": b.effective_cost,
                "min_effective_cost_zec": b.effective_cost_zec,
                "honest_overpayment_zats": b.honest_overpayment,
                "honest_overpayment_zec": b.honest_overpayment_zec,
                "harm_ratio": b.harm_ratio,
                "actions_per_block": b.actions_per_block,
                "fee_multiplier": b.fee_multiplier,
                "fee_bucket_jumps": b.fee_bucket_jumps,
                "fast_lane_open_blocks": b.fast_lane_open_blocks,
            })
            entry["defenses"] = {}
            for d in report.defenses:
                entry["defenses"][d.defense_name] = {
                    "survives": d.still_meets_target,
                    "harm_ratio": d.harm_ratio,
                    "fee_bucket_jumps": d.fee_bucket_jumps,
                    "effective_cost_zec": d.effective_cost_zec,
                }
        all_json.append(entry)

    # ---- Summary section ----
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append("| attack | cheapest cost (ZEC) | harm_ratio | defenses survived |")
    md_lines.append("|---|---|---|---|")
    for report in reports:
        b = report.opt.best
        if b:
            survived = sum(1 for d in report.defenses if d.still_meets_target)
            total = len(report.defenses)
            md_lines.append(f"| {report.opt.attack_name} | {b.effective_cost_zec:.4f} | "
                            f"{b.harm_ratio:.4f} | {survived}/{total} |")
        else:
            md_lines.append(f"| {report.opt.attack_name} | infeasible | — | — |")
    md_lines.append("")

    # Write outputs
    (out / "adversarial_report.md").write_text("\n".join(md_lines))
    with open(out / "adversarial_results.json", "w") as f:
        json.dump(all_json, f, indent=2)

    # Write grid CSV for each attack
    for report in reports:
        name = report.opt.attack_name
        rows = report.opt.all_evaluated
        if not rows:
            continue
        csv_path = out / f"grid_{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["actions_per_block", "fee_multiplier", "effective_cost",
                             "effective_cost_zec", "harm_ratio", "fee_jumps",
                             "fl_open", "fl_flaps", "meets_target"])
            for r in rows:
                writer.writerow([r.actions_per_block, r.fee_multiplier, r.effective_cost,
                                 r.effective_cost_zec, r.harm_ratio, r.fee_bucket_jumps,
                                 r.fast_lane_open_blocks, r.fast_lane_flaps, r.meets_target])

    print(f"\n  Report: {out / 'adversarial_report.md'}")
    print(f"  JSON:   {out / 'adversarial_results.json'}")
    return reports
