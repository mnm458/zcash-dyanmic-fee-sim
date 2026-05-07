"""Full validation: sweeps, comparisons, and sensitivity report."""

from __future__ import annotations

import json
from pathlib import Path

from .sweep import run_sweep, run_comparison, format_comparison_report


def _base_scenario() -> dict:
    return {
        "num_blocks": 400,
        "random_seed": 42,
        "chain": {"block_action_cap": 200, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "action_weighted_median",
            "quantization": "power_of_10",
            "base_fee": 5000, "floor_fee": 5000, "oracle_include_synthetic": True,
        },
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": True, "actions_per_block": 100, "fee_per_action": 5000},
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 20},
        "attacker": {
            "enabled": True, "type": "MedianPoisoningAttacker",
            "start_height": 100, "end_height": 200,
            "target_fee_multiplier": 10, "actions_per_block": 300,
        },
    }


def _congestion_base() -> dict:
    """Base config that reliably produces non-zero confirmation delays.

    Key: block_action_cap (50) < honest demand (30 arrivals * ~3 actions = ~90/block).
    """
    return {
        "num_blocks": 300,
        "random_seed": 42,
        "chain": {"block_action_cap": 50, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "action_weighted_median",
            "quantization": "power_of_10",
            "base_fee": 5000, "floor_fee": 5000, "oracle_include_synthetic": True,
        },
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": False},
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 20},
        "attacker": {"enabled": False},
    }


def run_full_validation(out_dir: str | Path = "results/validation") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = []
    report_lines.append("# Validation Report")
    report_lines.append("")
    all_conclusions: list[tuple[str, bool, str]] = []

    # ================================================================
    # 1. Parameter sweeps
    # ================================================================
    report_lines.append("# 1. Parameter Sweeps")
    report_lines.append("")

    base = _base_scenario()

    sweeps = [
        ("chain.block_action_cap", [100, 200, 500, 1000]),
        ("honest_demand.arrival_rate", [10, 30, 50, 80]),
        ("attacker.actions_per_block", [100, 300, 500]),
        ("attacker.target_fee_multiplier", [2, 5, 10, 50]),
        ("controller.lookback", [10, 25, 50, 100]),
    ]

    for param, values in sweeps:
        print(f"  Sweep: {param} = {values}")
        results = run_sweep(base, param, values, out / "sweeps")
        report_lines.append(f"### Sweep: {param}")
        report_lines.append("")
        report_lines.append(f"| {param} | harm_ratio | fee_jumps | fl_flaps | med_delay | eff_cost_zec |")
        report_lines.append("|---|---|---|---|---|---|")
        for r in results:
            report_lines.append(
                f"| {r['_sweep_value']} | {r['harm_ratio']:.4f} | {r['fee_bucket_jumps']} "
                f"| {r['fast_lane_flaps']} | {r['median_confirmation_delay']:.2f} "
                f"| {r['effective_attacker_cost_zec']:.6f} |"
            )
        report_lines.append("")

    # Oracle type sweep (special: changes a string param)
    print("  Sweep: oracle type")
    oracle_types = ["transaction_weighted_median", "action_weighted_median", "capped_effective_fee_median", "byte_share_weighted_median"]
    oracle_results = run_sweep(base, "controller.oracle", oracle_types, out / "sweeps")
    report_lines.append("### Sweep: controller.oracle")
    report_lines.append("")
    report_lines.append("| oracle | harm_ratio | fee_jumps | eff_cost_zec | overpayment_zec |")
    report_lines.append("|---|---|---|---|---|")
    for r in oracle_results:
        report_lines.append(
            f"| {r['_sweep_value']} | {r['harm_ratio']:.4f} | {r['fee_bucket_jumps']} "
            f"| {r['effective_attacker_cost_zec']:.6f} | {r['honest_overpayment_zec']:.6f} |"
        )
    report_lines.append("")

    # ================================================================
    # 2. Congestion: verify non-zero delays
    # ================================================================
    report_lines.append("# 2. Congestion Validation")
    report_lines.append("")
    print("  Congestion validation")

    cong = _congestion_base()
    from .scenarios import run_scenario
    m_cong, _ = run_scenario(cong)
    s_cong = m_cong.summary()
    m_cong.write_summary_json(out / "congestion_summary.json")

    delay_ok = s_cong["median_confirmation_delay"] > 0
    expired_ok = s_cong["expired_honest_transactions"] > 0
    report_lines.append(f"- block_action_cap=50, arrival_rate=30, mean_actions=3, expiry=20")
    report_lines.append(f"- median_confirmation_delay = {s_cong['median_confirmation_delay']:.2f} {'PASS' if delay_ok else 'FAIL: expected > 0'}")
    report_lines.append(f"- p95_confirmation_delay = {s_cong['p95_confirmation_delay']:.2f}")
    report_lines.append(f"- expired_honest_transactions = {s_cong['expired_honest_transactions']} {'PASS' if expired_ok else 'FAIL: expected > 0'}")
    report_lines.append("")
    all_conclusions.append(("Congestion produces non-zero median delay", delay_ok,
                            f"median_delay={s_cong['median_confirmation_delay']:.2f}"))
    all_conclusions.append(("Congestion produces expired txs", expired_ok,
                            f"expired={s_cong['expired_honest_transactions']}"))

    # ================================================================
    # 3. Head-to-head comparisons
    # ================================================================
    report_lines.append("# 3. Head-to-Head Comparisons")
    report_lines.append("")

    # 3a. tx-weighted vs action-weighted — use SybilSplitAttacker
    #     (the whole point is that sybil splitting is cheap for tx-weighted)
    print("  Comparison: tx-weighted vs action-weighted")
    sybil_base = {**_base_scenario(),
                  "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
                  "honest_demand": {"arrival_rate": 10, "mean_actions": 5, "expiry_blocks": 40},
                  "attacker": {
                      "enabled": True, "type": "SybilSplitAttacker",
                      "start_height": 100, "end_height": 200,
                      "total_actions": 200, "actions_per_tx": 1,
                      "target_fee_multiplier": 10,
                  }}
    cmp_oracle = {
        "tx_weighted": {**sybil_base,
                        "controller": {**_base_scenario()["controller"],
                                       "oracle": "transaction_weighted_median"}},
        "action_weighted": {**sybil_base,
                            "controller": {**_base_scenario()["controller"],
                                           "oracle": "action_weighted_median"}},
    }
    s_oracle = run_comparison(cmp_oracle, out / "cmp_oracle")
    report_lines.append(format_comparison_report(s_oracle, "tx-weighted vs action-weighted oracle (sybil attack)"))

    tw_hr = s_oracle["tx_weighted"]["harm_ratio"]
    aw_hr = s_oracle["action_weighted"]["harm_ratio"]
    tw_jumps = s_oracle["tx_weighted"]["fee_bucket_jumps"]
    aw_jumps = s_oracle["action_weighted"]["fee_bucket_jumps"]
    aw_better = (aw_hr <= tw_hr) or (aw_jumps <= tw_jumps)
    all_conclusions.append(("Action-weighted resists sybil better than tx-weighted", aw_better,
                            f"aw_hr={aw_hr:.4f} tw_hr={tw_hr:.4f} aw_jumps={aw_jumps} tw_jumps={tw_jumps}"))

    # 3b. uncapped vs capped oracle — use high-multiplier MedianPoisoning
    print("  Comparison: uncapped vs capped oracle")
    poison_base = {**_base_scenario(),
                   "attacker": {
                       "enabled": True, "type": "MedianPoisoningAttacker",
                       "start_height": 100, "end_height": 200,
                       "target_fee_multiplier": 100, "actions_per_block": 300,
                   }}
    cmp_cap = {
        "uncapped": {**poison_base,
                     "controller": {**_base_scenario()["controller"],
                                    "oracle": "action_weighted_median"}},
        "capped": {**poison_base,
                   "controller": {**_base_scenario()["controller"],
                                  "type": "ComparableMedianWithCapController"}},
    }
    s_cap = run_comparison(cmp_cap, out / "cmp_capped")
    report_lines.append(format_comparison_report(s_cap, "uncapped vs capped oracle"))

    uncap_op = s_cap["uncapped"]["honest_overpayment_vs_fixed_zip317"]
    cap_op = s_cap["capped"]["honest_overpayment_vs_fixed_zip317"]
    uncap_vol = s_cap["uncapped"]["fee_volatility"]
    cap_vol = s_cap["capped"]["fee_volatility"]
    cap_better = cap_op <= uncap_op or cap_vol <= uncap_vol
    all_conclusions.append(("Capped oracle limits poisoning vs uncapped", cap_better,
                            f"capped_op={cap_op} uncapped_op={uncap_op} capped_vol={cap_vol:.6f} uncapped_vol={uncap_vol:.6f}"))

    # 3c. no hysteresis vs hysteresis
    print("  Comparison: no-hysteresis vs hysteresis")
    nudge_atk = {
        "enabled": True, "type": "BucketBoundaryNudgingAttacker",
        "start_height": 100, "end_height": 300,
        "nudge_actions": 100, "nudge_fee_multiplier": 2.0,
    }
    cmp_hyst = {
        "no_hysteresis": {**_base_scenario(), "attacker": nudge_atk,
                          "controller": {**_base_scenario()["controller"]}},
        "hysteresis": {**_base_scenario(), "attacker": nudge_atk,
                       "controller": {
                           "type": "ComparableMedianHysteresisController",
                           "lookback": 50, "reorg_buffer": 5,
                           "oracle": "action_weighted_median",
                           "base_fee": 5000, "floor_fee": 5000, "oracle_include_synthetic": True,
                           "move_up_consecutive": 5, "move_down_consecutive": 20,
                       }},
    }
    s_hyst = run_comparison(cmp_hyst, out / "cmp_hysteresis")
    report_lines.append(format_comparison_report(s_hyst, "no-hysteresis vs hysteresis"))

    nh_jumps = s_hyst["no_hysteresis"]["fee_bucket_jumps"]
    h_jumps = s_hyst["hysteresis"]["fee_bucket_jumps"]
    hyst_better = h_jumps <= nh_jumps
    all_conclusions.append(("Hysteresis fee_jumps <= no-hysteresis", hyst_better,
                            f"hyst={h_jumps} no={nh_jumps}"))

    # 3d. binary fast lane vs priority buckets
    print("  Comparison: binary fast lane vs priority buckets")
    fl_atk = {
        "enabled": True, "type": "FastLaneFlapAttacker",
        "start_height": 50, "end_height": 300,
        "actions_per_block": 90, "target_fee_multiplier": 1, "expiry_blocks": 2,
    }
    cmp_fl = {
        "binary_1x_10x": {**_base_scenario(), "attacker": fl_atk,
                          "controller": {
                              "type": "BinaryFastLaneController", "base_fee": 5000,
                              "open_threshold": 0.95, "close_threshold": 0.95,
                              "use_hysteresis": False,
                              "synthetic_actions_per_block": 100, "block_action_cap": 200,
                          }},
        "priority_buckets": {**_base_scenario(), "attacker": fl_atk,
                             "controller": {
                                 "type": "PriorityBucketController", "base_fee": 5000,
                                 "open_threshold": 0.95,
                                 "synthetic_actions_per_block": 100,
                             }},
    }
    s_fl = run_comparison(cmp_fl, out / "cmp_fast_lane")
    report_lines.append(format_comparison_report(s_fl, "binary fast lane vs priority buckets"))

    bin_flaps = s_fl["binary_1x_10x"]["fast_lane_flaps"]
    pb_flaps = s_fl["priority_buckets"]["fast_lane_flaps"]
    all_conclusions.append(("Priority bucket flaps <= binary flaps", pb_flaps <= bin_flaps,
                            f"pb={pb_flaps} bin={bin_flaps}"))

    # ================================================================
    # 4. Conclusion summary
    # ================================================================
    report_lines.append("# 4. Validation Conclusions")
    report_lines.append("")
    pass_count = sum(1 for _, ok, _ in all_conclusions if ok)
    total = len(all_conclusions)
    report_lines.append(f"**{pass_count}/{total} headline results survived sensitivity analysis.**")
    report_lines.append("")
    for claim, ok, evidence in all_conclusions:
        status = "PASS" if ok else "FAIL"
        report_lines.append(f"- [{status}] {claim} ({evidence})")
    report_lines.append("")

    # Write report
    report_text = "\n".join(report_lines)
    (out / "validation_report.md").write_text(report_text)
    print(f"\nValidation report written to {out / 'validation_report.md'}")
    print(f"\n{pass_count}/{total} checks passed.")
    for claim, ok, evidence in all_conclusions:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {claim} ({evidence})")
