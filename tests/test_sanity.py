"""Sanity tests for unit scales, cost breakdown, and congestion."""

from zfeesim.oracle import ZATS_PER_ZEC, zats_to_zec
from zfeesim.scenarios import run_scenario


def _base(**over):
    cfg = {
        "num_blocks": 200,
        "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {"type": "FixedZip317Controller", "marginal_fee": 5000},
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": False},
        "honest_demand": {"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": False},
    }
    for k, v in over.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ---- Unit scale sanity ----

def test_zats_per_zec_constant():
    assert ZATS_PER_ZEC == 100_000_000


def test_zats_to_zec_conversion():
    assert zats_to_zec(100_000_000) == 1.0
    assert zats_to_zec(5000) == 0.00005
    assert zats_to_zec(0) == 0.0


def test_marginal_fee_is_5000_zats():
    """ZIP-317 marginal fee = 5000 zats = 0.00005 ZEC."""
    assert zats_to_zec(5000) == 0.00005


def test_conventional_fee_2_actions():
    """2-action tx conventional fee = 5000 * 2 = 10000 zats = 0.0001 ZEC."""
    from zfeesim.zip317 import conventional_fee
    from zfeesim.tx import make_tx
    from zfeesim.types import TxKind
    tx = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                 logical_actions=2, byte_size=500, fee_paid=10000)
    assert conventional_fee(tx, 5000) == 10000
    assert zats_to_zec(10000) == 0.0001


def test_tx_fee_paid_scale():
    """A 3-action tx paying 1x marginal = 15000 zats = 0.00015 ZEC."""
    from zfeesim.tx import make_tx
    from zfeesim.types import TxKind
    tx = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                 logical_actions=3, byte_size=750, fee_paid=15000)
    assert tx.fee_per_action == 5000.0
    assert zats_to_zec(tx.fee_paid) == 0.00015


def test_summary_zec_fields_present():
    """Summary must include ZEC-equivalent outputs."""
    m, _ = run_scenario(_base())
    s = m.summary()
    assert "effective_attacker_cost_zec" in s
    assert "honest_overpayment_zec" in s
    assert "honest_total_fee_zec" in s
    assert isinstance(s["effective_attacker_cost_zec"], float)
    assert isinstance(s["honest_total_fee_zec"], float)


def test_honest_total_fee_zec_plausible():
    """200 blocks * ~20 txs * ~3 actions * 5000 zats ≈ 60M zats ≈ 0.6 ZEC."""
    m, _ = run_scenario(_base())
    s = m.summary()
    zec = s["honest_total_fee_zec"]
    # Poisson mean ~20, actions ~3, fee ~5000 -> ~60M zats ~ 0.6 ZEC
    assert 0.1 < zec < 5.0, f"honest_total_fee_zec={zec} outside plausible range"


# ---- Cost breakdown ----

def test_cost_split_no_attacker():
    """Without attacker, all attacker cost fields are zero."""
    m, _ = run_scenario(_base())
    s = m.summary()
    assert s["attacker_nominal_fee_paid"] == 0
    assert s["miner_self_nominal_fee_paid"] == 0
    assert s["miner_recovered_fee"] == 0
    assert s["effective_attacker_cost"] == 0
    assert s["harm_ratio"] == 0


def test_cost_split_with_attacker():
    """With an attacker, nominal fee should be positive."""
    cfg = _base(attacker={
        "enabled": True, "type": "BurstSpamAttacker",
        "start_height": 50, "end_height": 60,
        "actions_per_block": 200, "target_fee_multiplier": 5,
    })
    m, _ = run_scenario(cfg)
    s = m.summary()
    assert s["attacker_nominal_fee_paid"] > 0
    assert s["effective_attacker_cost"] == s["attacker_nominal_fee_paid"]  # no miner self-dealing
    assert s["miner_recovered_fee"] == 0


def test_cost_split_miner_self_dealing():
    """Miner self-dealing: recovered fee should be fee_recovery_rate * nominal."""
    cfg = _base(attacker={
        "enabled": True, "type": "MinerSelfDealingAttacker",
        "start_height": 50, "end_height": 60,
        "actions_per_block": 100, "target_fee_multiplier": 3,
        "fee_recovery_rate": 0.8,
    })
    m, _ = run_scenario(cfg)
    s = m.summary()
    assert s["miner_self_nominal_fee_paid"] > 0
    expected_recovered = int(s["miner_self_nominal_fee_paid"] * 0.8)
    assert s["miner_recovered_fee"] == expected_recovered
    assert s["effective_attacker_cost"] == s["miner_self_nominal_fee_paid"] - expected_recovered


# ---- Congestion fix: non-zero delays ----

def test_congestion_produces_nonzero_delays():
    """When demand >> capacity, median delay must be > 0."""
    cfg = _base(
        chain={"block_action_cap": 50, "block_byte_cap": 2_000_000},
        honest_demand={"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 20},
    )
    m, _ = run_scenario(cfg)
    s = m.summary()
    assert s["median_confirmation_delay"] > 0, f"Expected nonzero delay, got {s['median_confirmation_delay']}"
    assert s["p95_confirmation_delay"] >= s["median_confirmation_delay"]


def test_congestion_produces_expired_txs():
    """When demand >> capacity and expiry is short, txs must expire."""
    cfg = _base(
        num_blocks=300,
        chain={"block_action_cap": 50, "block_byte_cap": 2_000_000},
        honest_demand={"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 15},
    )
    m, _ = run_scenario(cfg)
    s = m.summary()
    assert s["expired_honest_transactions"] > 0, f"Expected expired txs, got 0"


def test_attacker_cost_zec_plausible():
    """Attacker cost in ZEC should be reasonable for a 10-block burst."""
    cfg = _base(
        attacker={
            "enabled": True, "type": "BurstSpamAttacker",
            "start_height": 50, "end_height": 60,
            "actions_per_block": 500, "target_fee_multiplier": 10,
        }
    )
    m, _ = run_scenario(cfg)
    s = m.summary()
    cost_zec = s["effective_attacker_cost_zec"]
    # 10 blocks * 500 actions * 50000 fee/action = 250M zats = 2.5 ZEC
    assert 0.1 < cost_zec < 50.0, f"attacker_cost_zec={cost_zec} outside plausible range"


# ---- Oracle variant: byte_share is genuinely different ----

def test_byte_share_oracle_differs_from_action_weighted():
    """byte_share_weighted_median should produce different results from action_weighted
    when tx byte sizes are not proportional to logical actions."""
    from zfeesim.oracle import action_weighted_median_fee, byte_share_weighted_median_fee
    from zfeesim.tx import make_tx
    from zfeesim.types import Block, TxKind

    # tx A: 1 action, 1000 bytes, fee=50000 (50000 fpa)
    # tx B: 10 actions, 100 bytes, fee=50000 (5000 fpa)
    tx_a = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                   logical_actions=1, byte_size=1000, fee_paid=50000)
    tx_b = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                   logical_actions=10, byte_size=100, fee_paid=50000)
    blocks = [Block(height=0, txs=[tx_a, tx_b], marginal_fee=5000)]

    aw = action_weighted_median_fee(blocks)
    bw = byte_share_weighted_median_fee(blocks)

    # action-weighted: 1 unit at 50000, 10 units at 5000 -> median = 5000
    # byte-weighted: 1000 bytes at 50000, 100 bytes at 5000 -> median = 50000
    assert aw != bw, f"Expected different results, both returned {aw}"
