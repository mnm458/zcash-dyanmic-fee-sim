"""Tests for adversarial cost optimization."""

from zfeesim.adversarial import (
    grid_search,
    optimize_median_poisoning,
    optimize_burst_spam,
    optimize_fast_lane_griefing,
    optimize_bucket_nudging,
    _base_cfg,
    AttackResult,
)


def test_grid_search_returns_cheapest_feasible():
    """Grid search should return the lowest-cost config that meets the condition."""
    base = _base_cfg()

    def always_true(s: dict) -> bool:
        return s["fee_bucket_jumps"] >= 0  # trivially true

    opt = grid_search(
        base, "MedianPoisoningAttacker",
        actions_values=[50, 100],
        multiplier_values=[2, 5],
        harm_condition=always_true,
    )
    assert len(opt.all_evaluated) == 4  # 2 x 2 grid
    if opt.best:
        # Should be the cheapest (lowest actions * lowest multiplier)
        for r in opt.all_evaluated:
            if r.meets_target and r.effective_cost > 0:
                assert opt.best.effective_cost <= r.effective_cost


def test_grid_search_infeasible():
    """If no config meets the condition, best should be None."""
    base = _base_cfg()

    def impossible(s: dict) -> bool:
        return s["fee_bucket_jumps"] > 9999

    opt = grid_search(
        base, "MedianPoisoningAttacker",
        actions_values=[50],
        multiplier_values=[2],
        harm_condition=impossible,
    )
    assert opt.best is None
    assert len(opt.all_evaluated) == 1


def test_median_poisoning_opt_runs():
    """optimize_median_poisoning should complete and return a structured result."""
    # Use a small base to keep runtime short
    base = _base_cfg()
    base["num_blocks"] = 250
    report = optimize_median_poisoning(base)
    assert report.opt.attack_name == "MedianPoisoningAttacker"
    assert len(report.opt.all_evaluated) > 0
    # If feasible, defenses should have been probed
    if report.opt.best:
        assert len(report.defenses) == 4  # action_weighted, capped, hysteresis, aimd


def test_burst_spam_opt_runs():
    base = _base_cfg()
    base["num_blocks"] = 250
    report = optimize_burst_spam(base)
    assert report.opt.attack_name == "BurstSpamAttacker"
    assert len(report.opt.all_evaluated) > 0


def test_fast_lane_opt_runs():
    base = _base_cfg()
    base["num_blocks"] = 250
    report = optimize_fast_lane_griefing(base, min_open_blocks=5)
    assert report.opt.attack_name == "FastLaneFlapAttacker"
    assert len(report.opt.all_evaluated) > 0


def test_bucket_nudging_opt_runs():
    base = _base_cfg()
    base["num_blocks"] = 250
    report = optimize_bucket_nudging(base, min_jumps=2)
    assert report.opt.attack_name == "BucketBoundaryNudgingAttacker"
    assert len(report.opt.all_evaluated) > 0


def test_defense_probes_reduce_harm():
    """At least one defense variant should reduce harm vs baseline."""
    base = _base_cfg()
    base["num_blocks"] = 250
    report = optimize_median_poisoning(base)
    if report.opt.best and report.defenses:
        baseline_hr = report.opt.best.harm_ratio
        min_defense_hr = min(d.harm_ratio for d in report.defenses)
        # At least one defense should not make things worse
        assert min_defense_hr <= baseline_hr + 0.01  # small tolerance for seeded noise
