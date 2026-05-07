"""Regression tests for synthetic tx inclusion in the oracle.

These tests verify the critical fix: synthetic transactions must
participate in the oracle median when oracle_include_synthetic=true
(the proposal-faithful default).
"""

from zfeesim.oracle import (
    action_weighted_median_fee,
    transaction_weighted_median_fee,
    oracle_sample_breakdown,
)
from zfeesim.tx import make_tx
from zfeesim.types import Block, TxKind
from zfeesim.scenarios import run_scenario


def _block(height, tx_specs):
    """tx_specs: list of (kind, actions, fee_paid)"""
    txs = [
        make_tx(kind=kind, created_height=height, expiry_height=height + 40,
                logical_actions=actions, byte_size=actions * 250, fee_paid=fee)
        for kind, actions, fee in tx_specs
    ]
    return Block(height=height, txs=txs, marginal_fee=5000)


# ---- Core inclusion/exclusion ----

def test_synthetic_included_when_flag_true():
    """With include_synthetic=True, synthetic txs enter the median sample."""
    blocks = [_block(0, [
        (TxKind.HONEST, 3, 15000),     # 5000 fpa
        (TxKind.SYNTHETIC, 100, 500000),  # 5000 fpa
        (TxKind.ATTACKER, 50, 5000000),   # 100000 fpa
    ])]
    bd = oracle_sample_breakdown(blocks, include_synthetic=True)
    assert bd["synthetic_actions_in_oracle"] == 100
    assert bd["total_actions_in_oracle"] == 153  # 3 + 100 + 50


def test_synthetic_excluded_when_flag_false():
    """With include_synthetic=False, synthetic txs are skipped."""
    blocks = [_block(0, [
        (TxKind.HONEST, 3, 15000),
        (TxKind.SYNTHETIC, 100, 500000),
        (TxKind.ATTACKER, 50, 5000000),
    ])]
    bd = oracle_sample_breakdown(blocks, include_synthetic=False)
    assert bd["synthetic_actions_in_oracle"] == 0
    assert bd["total_actions_in_oracle"] == 53  # 3 + 50


def test_included_synthetic_anchors_median_down():
    """Including synthetic low-fee actions should pull median below attacker fee."""
    blocks = [_block(0, [
        (TxKind.HONEST, 3, 15000),        # 5000 fpa, weight 3
        (TxKind.SYNTHETIC, 100, 500000),   # 5000 fpa, weight 100
        (TxKind.ATTACKER, 50, 5000000),    # 100000 fpa, weight 50
    ])]
    med_included = action_weighted_median_fee(blocks, include_synthetic=True)
    med_excluded = action_weighted_median_fee(blocks, include_synthetic=False)

    # With synthetic: 103 actions at 5000 vs 50 at 100000 -> median = 5000
    assert med_included == 5000.0
    # Without synthetic: 3 at 5000 vs 50 at 100000 -> median = 100000
    assert med_excluded == 100000.0
    # The key invariant
    assert med_included < med_excluded


# ---- Burst scenario regression ----

def test_burst_spam_median_stable_with_synthetic():
    """10-block burst of 500 attacker actions should NOT move the action-weighted
    median when 100 synthetic actions/block anchor the bottom across 50 blocks.

    Math:
    - 50-block lookback window
    - 10 blocks contain 500 attacker actions at 100000 fpa
    - 50 blocks contain 100 synthetic actions at 5000 fpa
    - ~50 blocks contain ~90 honest actions at ~5000-10000 fpa
    - Total: ~4500 honest + 5000 synthetic at ~5000 + 5000 attacker at 100000
    - Sorted: 9500 actions at <=10000, then 5000 at 100000
    - Median is at action 7250 out of 14500 -> well within the honest+synthetic range
    """
    cfg = {
        "num_blocks": 250,
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
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {
            "enabled": True, "type": "BurstSpamAttacker",
            "start_height": 100, "end_height": 110,
            "target_fee_multiplier": 10, "actions_per_block": 500,
        },
    }
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()

    # With synthetic anchoring, the fee bucket should NOT jump
    assert s["public_fee_bucket_final"] <= 10000, \
        f"Fee jumped to {s['public_fee_bucket_final']} — synthetic anchoring failed"
    assert s["fee_bucket_jumps"] <= 1, \
        f"Got {s['fee_bucket_jumps']} jumps — expected <=1 with synthetic anchoring"


def test_excluded_synthetic_allows_burst_to_jump():
    """Same burst scenario but with synthetic excluded — fee SHOULD jump."""
    cfg = {
        "num_blocks": 250,
        "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "action_weighted_median",
            "quantization": "power_of_10",
            "base_fee": 5000, "floor_fee": 5000,
            "oracle_include_synthetic": False,
        },
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": True, "actions_per_block": 100, "fee_per_action": 5000},
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {
            "enabled": True, "type": "BurstSpamAttacker",
            "start_height": 100, "end_height": 110,
            "target_fee_multiplier": 10, "actions_per_block": 500,
        },
    }
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()

    # Without anchoring, fee should have jumped
    assert s["public_fee_bucket_final"] >= 100000, \
        f"Fee only reached {s['public_fee_bucket_final']} — expected escalation without synthetic"


def test_oracle_breakdown_in_summary_csv():
    """The per-block records should contain oracle sample breakdown fields."""
    cfg = {
        "num_blocks": 100,
        "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "action_weighted_median",
            "base_fee": 5000, "floor_fee": 5000,
            "oracle_include_synthetic": True,
        },
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": True, "actions_per_block": 100, "fee_per_action": 5000},
        "honest_demand": {"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": False},
    }
    metrics, _ = run_scenario(cfg)

    # After lookback + reorg_buffer blocks, oracle should have data
    late = metrics.records[-1]
    assert late.total_actions_in_oracle > 0
    assert late.synthetic_actions_in_oracle > 0
    assert late.honest_actions_in_oracle > 0
