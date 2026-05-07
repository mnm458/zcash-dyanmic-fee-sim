"""Tests for audit correctness checks — verifiable without running full audit."""

from zfeesim.oracle import ZATS_PER_ZEC, zats_to_zec
from zfeesim.scenarios import run_scenario


def _attack_cfg(atk_type: str, **kw) -> dict:
    base = {
        "num_blocks": 250, "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {"type": "ComparableMedianController", "lookback": 50,
                       "reorg_buffer": 5, "oracle": "action_weighted_median",
                       "base_fee": 5000, "floor_fee": 5000},
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": True, "actions_per_block": 100, "fee_per_action": 5000},
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 20},
        "attacker": {"enabled": True, "type": atk_type, "start_height": 50,
                     "end_height": 100, **kw},
    }
    return base


# ---- Item 2: harm_ratio formula ----

def test_harm_ratio_formula():
    """harm_ratio must equal honest_overpayment / effective_attacker_cost."""
    cfg = _attack_cfg("BurstSpamAttacker", actions_per_block=200, target_fee_multiplier=5)
    m, _ = run_scenario(cfg)
    s = m.summary()
    if s["effective_attacker_cost"] > 0:
        expected = s["honest_overpayment_vs_fixed_zip317"] / s["effective_attacker_cost"]
        assert abs(s["harm_ratio"] - round(expected, 4)) < 0.001, \
            f"harm_ratio {s['harm_ratio']} != computed {expected:.4f}"


def test_harm_ratio_zero_without_attacker():
    cfg = _attack_cfg("BurstSpamAttacker", actions_per_block=0, target_fee_multiplier=1)
    cfg["attacker"]["enabled"] = False
    m, _ = run_scenario(cfg)
    assert m.summary()["harm_ratio"] == 0.0


# ---- Item 3: attacker_cost excludes synthetic and honest ----

def test_attacker_cost_excludes_synthetic():
    """Run with synthetic enabled but no attacker. Cost must be 0."""
    cfg = _attack_cfg("BurstSpamAttacker")
    cfg["attacker"]["enabled"] = False
    cfg["synthetic"]["enabled"] = True
    m, _ = run_scenario(cfg)
    s = m.summary()
    assert s["attacker_nominal_fee_paid"] == 0
    assert s["miner_self_nominal_fee_paid"] == 0
    assert s["effective_attacker_cost"] == 0
    # But synthetic should have been included
    assert any(r.synthetic_actions_included > 0 for r in m.records)


def test_attacker_cost_excludes_honest():
    """Honest fees must not appear in attacker cost."""
    cfg = _attack_cfg("BurstSpamAttacker", actions_per_block=100, target_fee_multiplier=5)
    m, _ = run_scenario(cfg)
    s = m.summary()
    # honest_total_fee should be positive
    assert s["honest_total_fee"] > 0
    # attacker cost should only reflect attacker txs
    assert s["attacker_nominal_fee_paid"] > 0
    assert s["effective_attacker_cost"] == s["attacker_nominal_fee_paid"]  # no miner self-dealing


# ---- Item 4: ZEC conversion ----

def test_zec_conversion_constant():
    assert ZATS_PER_ZEC == 100_000_000


def test_zec_conversion_in_summary():
    cfg = _attack_cfg("BurstSpamAttacker", actions_per_block=100, target_fee_multiplier=5)
    m, _ = run_scenario(cfg)
    s = m.summary()
    # Check each ZEC field matches zats / 100_000_000
    assert abs(s["effective_attacker_cost_zec"] - s["effective_attacker_cost"] / 1e8) < 1e-6
    assert abs(s["honest_overpayment_zec"] - s["honest_overpayment_vs_fixed_zip317"] / 1e8) < 1e-6
    assert abs(s["honest_total_fee_zec"] - s["honest_total_fee"] / 1e8) < 1e-6


# ---- Item 5: median poisoning vs bucket nudging distinctness ----

def test_median_poisoning_and_nudging_are_separate_types():
    """The two attackers use different parameter names and can produce different behavior."""
    from zfeesim.attackers import MedianPoisoningAttacker, BucketBoundaryNudgingAttacker
    import inspect

    mp_params = set(inspect.signature(MedianPoisoningAttacker.__init__).parameters.keys())
    bn_params = set(inspect.signature(BucketBoundaryNudgingAttacker.__init__).parameters.keys())

    # BucketBoundary uses nudge_actions + nudge_fee_multiplier
    assert "nudge_actions" in bn_params
    assert "nudge_fee_multiplier" in bn_params
    # MedianPoisoning uses actions_per_block + target_fee_multiplier
    assert "actions_per_block" in mp_params
    assert "target_fee_multiplier" in mp_params
    # Distinct parameter names
    assert "nudge_actions" not in mp_params


# ---- Item 6: AIMD defense doesn't make system unusable ----

def test_aimd_defense_still_confirms_txs():
    """Under AIMD, honest txs should still confirm without extreme delays."""
    cfg = _attack_cfg("BurstSpamAttacker", actions_per_block=200, target_fee_multiplier=5)
    cfg["controller"] = {
        "type": "AIMDBucketController", "base_fee": 5000,
        "alpha": 0.25, "beta": 0.90, "target_utilization": 0.70,
        "lower_utilization": 0.40, "increase_window": 5,
        "decrease_window": 20, "block_action_cap": 1000,
    }
    m, _ = run_scenario(cfg)
    s = m.summary()
    # AIMD should not cause mass expirations or extreme delays
    assert s["median_confirmation_delay"] <= 5
    assert s["expired_honest_transactions"] <= 50  # generous bound
    # Fee should still be in a reasonable range
    assert s["public_fee_bucket_final"] <= 100_000
