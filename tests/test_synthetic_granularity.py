"""Tests for synthetic demand granularity fix.

Verifies that granular synthetic txs enable partial capacity fill
and continuous displacement ratios, preventing the all-or-nothing
fast-lane artifact from the old atomic model.
"""

from zfeesim.synthetic import SyntheticPolicy
from zfeesim.types import TxKind
from zfeesim.scenarios import run_scenario


# ---- A. Granular generation ----

def test_granular_generates_many_txs():
    sp = SyntheticPolicy(actions_per_block=100, fee_per_action=5000,
                         granularity_mode="granular", tx_granularity_actions=1)
    txs = sp.generate(0, 5000)
    assert len(txs) == 100
    assert all(tx.logical_actions == 1 for tx in txs)
    assert all(tx.kind == TxKind.SYNTHETIC for tx in txs)
    assert sum(tx.logical_actions for tx in txs) == 100
    assert sum(tx.fee_paid for tx in txs) == 100 * 5000


def test_granular_5_actions_per_tx():
    sp = SyntheticPolicy(actions_per_block=100, fee_per_action=5000,
                         granularity_mode="granular", tx_granularity_actions=5)
    txs = sp.generate(0, 5000)
    assert len(txs) == 20
    assert all(tx.logical_actions == 5 for tx in txs)
    assert sum(tx.logical_actions for tx in txs) == 100


# ---- B. Remainder handling ----

def test_granular_remainder():
    sp = SyntheticPolicy(actions_per_block=103, fee_per_action=5000,
                         granularity_mode="granular", tx_granularity_actions=10)
    txs = sp.generate(0, 5000)
    assert len(txs) == 11  # 10 full + 1 remainder
    assert txs[-1].logical_actions == 3
    assert sum(tx.logical_actions for tx in txs) == 103


# ---- C. Atomic legacy mode ----

def test_atomic_generates_one_tx():
    sp = SyntheticPolicy(actions_per_block=100, fee_per_action=5000,
                         granularity_mode="atomic", tx_granularity_actions=1)
    txs = sp.generate(0, 5000)
    assert len(txs) == 1
    assert txs[0].logical_actions == 100


# ---- D. Displacement is action-based ----

def test_displacement_action_based():
    """Run a scenario and verify displacement ratio is computed from actions."""
    cfg = {
        "num_blocks": 100, "random_seed": 42,
        "chain": {"block_action_cap": 200, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {"type": "FixedZip317Controller", "marginal_fee": 5000},
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": True, "actions_per_block": 100, "fee_per_action": 5000,
                      "granularity_mode": "granular", "tx_granularity_actions": 1},
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": False},
    }
    m, _ = run_scenario(cfg)
    # With ~90 honest + 100 synthetic offered, cap=200, about 100+90=190 fit
    # So most synthetic should be included, displacement near 0
    for r in m.records[10:]:  # skip warmup
        if r.synthetic_actions_included > 0:
            expected_disp = 1.0 - (r.synthetic_actions_included / 100)
            assert abs(r.synthetic_displacement_ratio - expected_disp) < 0.01


# ---- E. Fast-lane threshold continuous ----

def test_fast_lane_threshold_continuous():
    """With granular synthetic, displacement should be continuous, not binary."""
    # 150 honest demand + 100 synthetic = 250, cap = 200
    # Only ~50 synthetic should fit -> displacement ≈ 0.5
    cfg = {
        "num_blocks": 100, "random_seed": 42,
        "chain": {"block_action_cap": 200, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {"type": "BinaryFastLaneController", "base_fee": 5000,
                        "open_threshold": 0.95, "close_threshold": 0.95,
                        "use_hysteresis": False,
                        "synthetic_actions_per_block": 100, "block_action_cap": 200},
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": True, "actions_per_block": 100, "fee_per_action": 5000,
                      "granularity_mode": "granular", "tx_granularity_actions": 1},
        "honest_demand": {"arrival_rate": 50, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": False},
    }
    m, _ = run_scenario(cfg)
    # Displacement should be intermediate, not always 0 or 1
    disps = [r.synthetic_displacement_ratio for r in m.records[10:]]
    non_binary = [d for d in disps if 0.01 < d < 0.99]
    assert len(non_binary) > 0, f"All displacements are binary: {set(disps)}"


# ---- F. Fast-lane attack harder under granular ----

def test_fast_lane_attack_harder_granular_vs_atomic():
    """The old 90-action attack should NOT open fast lane under granular synthetic."""
    base = {
        "num_blocks": 200, "random_seed": 42,
        "chain": {"block_action_cap": 200, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {"type": "BinaryFastLaneController", "base_fee": 5000,
                        "open_threshold": 0.95, "close_threshold": 0.95,
                        "use_hysteresis": False,
                        "synthetic_actions_per_block": 100, "block_action_cap": 200},
        "block_builder": {"mode": "highest_fee_per_action"},
        "honest_demand": {"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": True, "type": "FastLaneFlapAttacker",
                     "start_height": 50, "end_height": 150,
                     "actions_per_block": 90, "target_fee_multiplier": 1,
                     "expiry_blocks": 2},
    }

    # Atomic: should open fast lane (old behavior)
    cfg_atomic = {**base, "synthetic": {
        "enabled": True, "actions_per_block": 100, "fee_per_action": 5000,
        "granularity_mode": "atomic", "tx_granularity_actions": 100,
    }}
    m_atomic, _ = run_scenario(cfg_atomic)
    s_atomic = m_atomic.summary()

    # Granular: should NOT open fast lane (fixed behavior)
    cfg_granular = {**base, "synthetic": {
        "enabled": True, "actions_per_block": 100, "fee_per_action": 5000,
        "granularity_mode": "granular", "tx_granularity_actions": 1,
    }}
    m_granular, _ = run_scenario(cfg_granular)
    s_granular = m_granular.summary()

    # Atomic should have many fast-lane-open blocks
    assert s_atomic["fast_lane_open_blocks"] > 50

    # Granular should have far fewer (ideally zero)
    assert s_granular["fast_lane_open_blocks"] < s_atomic["fast_lane_open_blocks"]

    # Granular displacement should be much lower than 0.95
    avg_disp_attack = sum(
        r.synthetic_displacement_ratio for r in m_granular.records if 50 <= r.height < 150
    ) / 100
    assert avg_disp_attack < 0.95, f"Granular avg displacement {avg_disp_attack:.2f} >= 0.95"
