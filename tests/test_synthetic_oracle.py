"""Tests for synthetic participation in the oracle.

Verifies that:
  - Synthetic txs in blocks are included/excluded in oracle based on flag
  - Median-of-medians with adaptive synthetic anchors correctly
  - Per-block median includes synthetic samples from capacity math
  - Oracle breakdown fields are populated
"""

from zfeesim.oracle import (
    action_weighted_median_fee,
    median_of_medians_fee,
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
    blocks = [_block(0, [
        (TxKind.HONEST, 3, 15000),
        (TxKind.SYNTHETIC, 100, 500000),
        (TxKind.ATTACKER, 50, 5000000),
    ])]
    bd = oracle_sample_breakdown(blocks, include_synthetic=True)
    assert bd["synthetic_actions_in_oracle"] == 100
    assert bd["total_actions_in_oracle"] == 153


def test_synthetic_excluded_when_flag_false():
    blocks = [_block(0, [
        (TxKind.HONEST, 3, 15000),
        (TxKind.SYNTHETIC, 100, 500000),
        (TxKind.ATTACKER, 50, 5000000),
    ])]
    bd = oracle_sample_breakdown(blocks, include_synthetic=False)
    assert bd["synthetic_actions_in_oracle"] == 0
    assert bd["total_actions_in_oracle"] == 53


# ---- Median-of-medians with synthetic anchoring ----

def test_median_of_medians_concentrated_attack():
    """Key test from audit: 2 attack blocks with 500 high-fee actions in a
    50-block window. Median-of-medians: per-block median, then window median.

    48 normal blocks: median = 5000 (honest + synthetic at floor)
    2 attack blocks: median = 50000 (attacker dominates that block's median)
    Window median of [5000]*48 + [50000]*2 = 5000

    Attack has no effect on the window median.
    """
    blocks = []
    for h in range(50):
        if h in (25, 26):  # 2 attack blocks
            specs = (
                [(TxKind.HONEST, 3, 15000)] * 10 +       # 10 honest at 5000 fpa
                [(TxKind.SYNTHETIC, 3, 15000)] * 50 +     # 50 synthetic at 5000 fpa
                [(TxKind.ATTACKER, 1, 50000)] * 500       # 500 attacker at 50000 fpa
            )
        else:  # normal blocks
            specs = (
                [(TxKind.HONEST, 3, 15000)] * 30 +        # 30 honest at 5000 fpa
                [(TxKind.SYNTHETIC, 3, 15000)] * 100       # 100 synthetic at 5000 fpa
            )
        blocks.append(_block(h, specs))

    result = median_of_medians_fee(blocks, include_synthetic=True)
    assert result == 5000.0, f"Expected 5000, got {result}. Attack should not move median-of-medians."


def test_median_of_medians_majority_attack():
    """If attacker controls >50% of blocks' medians, the window median moves."""
    blocks = []
    for h in range(50):
        if h < 30:  # 30 attack blocks
            specs = [(TxKind.ATTACKER, 1, 50000)] * 100  # median = 50000
        else:  # 20 normal blocks
            specs = [(TxKind.HONEST, 3, 15000)] * 30 + [(TxKind.SYNTHETIC, 3, 15000)] * 50
        blocks.append(_block(h, specs))

    result = median_of_medians_fee(blocks, include_synthetic=True)
    assert result == 50000.0, f"Expected 50000, got {result}. Majority attack should move median."


# ---- Full simulation: adaptive synthetic with median-of-medians ----

def test_burst_does_not_move_oracle_with_adaptive_synthetic():
    """End-to-end: 10-block burst of 500 high-fee actions should NOT move the
    median-of-medians oracle when adaptive synthetic anchors each block.

    Even in attack blocks, adaptive synthetic is computed from remaining byte
    capacity and appended to the block. If the attacker uses 500*250=125000 bytes
    of 2000000 capacity, remaining = 1875000 bytes, so k_i = 2500 synthetic
    samples at 5000 fpa are appended. The per-block median stays at 5000.
    """
    cfg = {
        "num_blocks": 250, "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "median_of_medians",
            "oracle_include_synthetic": True,
            "quantization": "round10",
            "base_fee": 5000, "floor_fee": 5000,
        },
        "block_builder": {"mode": "zip317_weighted_random"},
        "synthetic": {
            "enabled": True, "fee_per_action": 5000,
            "granularity_mode": "adaptive", "median_tx_actions": 3,
        },
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {
            "enabled": True, "type": "BurstSpamAttacker",
            "start_height": 100, "end_height": 110,
            "target_fee_multiplier": 10, "actions_per_block": 500,
        },
    }
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()
    # Fee should not jump beyond the baseline round10 bucket
    assert s["public_fee_bucket_final"] <= 10000
    # Oracle should stay at floor
    assert s["raw_oracle_fee_final"] == 5000.0


def test_oracle_breakdown_populated():
    cfg = {
        "num_blocks": 100, "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "median_of_medians",
            "base_fee": 5000, "floor_fee": 5000,
            "oracle_include_synthetic": True,
        },
        "block_builder": {"mode": "zip317_weighted_random"},
        "synthetic": {
            "enabled": True, "fee_per_action": 5000,
            "granularity_mode": "adaptive", "median_tx_actions": 3,
        },
        "honest_demand": {"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": False},
    }
    metrics, _ = run_scenario(cfg, compute_baseline=False)
    late = metrics.records[-1]
    assert late.total_actions_in_oracle > 0
    assert late.synthetic_actions_in_oracle > 0
    assert late.honest_actions_in_oracle > 0
