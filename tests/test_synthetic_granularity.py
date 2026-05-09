"""Tests for synthetic demand modeling.

Covers:
  - Adaptive mode: synthetic computed from remaining byte capacity after block building
  - Synthetic never enters mempool, never displaced by block builder
  - Synthetic count adapts per block (full block = 0 synthetic, half-full = many)
  - Legacy modes (granular, atomic) still work for comparison
  - Synthetic always participates in oracle regardless of block fullness
"""

from zfeesim.synthetic import SyntheticPolicy
from zfeesim.tx import make_tx
from zfeesim.types import Block, Tx, TxKind
from zfeesim.scenarios import run_scenario


# -----------------------------------------------------------------------
# Unit tests: SyntheticPolicy
# -----------------------------------------------------------------------

def test_adaptive_empty_block_fills_capacity():
    """Empty block -> synthetic fills all remaining byte capacity."""
    sp = SyntheticPolicy(enabled=True, fee_per_action=5000,
                         granularity_mode="adaptive",
                         block_byte_cap=2_000_000, median_tx_actions=3,
                         byte_size_per_action=250)
    block = Block(height=0, txs=[], marginal_fee=5000)
    synth = sp.generate_for_block(0, block)
    # median_tx_bytes = 250 * 3 = 750
    # k_i = floor(2_000_000 / 750) = 2666
    assert len(synth) == 2666
    assert all(tx.kind == TxKind.SYNTHETIC for tx in synth)
    assert all(tx.logical_actions == 3 for tx in synth)
    assert all(tx.fee_per_action == 5000 for tx in synth)


def test_adaptive_full_block_zero_synthetic():
    """Block using all byte capacity -> 0 synthetic samples."""
    sp = SyntheticPolicy(enabled=True, fee_per_action=5000,
                         granularity_mode="adaptive",
                         block_byte_cap=2_000_000, median_tx_actions=3,
                         byte_size_per_action=250)
    # Fill block with real txs consuming all 2MB
    real_tx = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                      logical_actions=8000, byte_size=2_000_000, fee_paid=40_000_000)
    block = Block(height=0, txs=[real_tx], marginal_fee=5000)
    synth = sp.generate_for_block(0, block)
    assert len(synth) == 0


def test_adaptive_half_full_block():
    """Block using half capacity -> synthetic fills remaining half."""
    sp = SyntheticPolicy(enabled=True, fee_per_action=5000,
                         granularity_mode="adaptive",
                         block_byte_cap=2_000_000, median_tx_actions=3,
                         byte_size_per_action=250)
    real_tx = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                      logical_actions=4000, byte_size=1_000_000, fee_paid=20_000_000)
    block = Block(height=0, txs=[real_tx], marginal_fee=5000)
    synth = sp.generate_for_block(0, block)
    # remaining = 1_000_000, median_tx_bytes = 750, k_i = 1333
    assert len(synth) == 1333


def test_adaptive_returns_nothing_for_mempool():
    """Adaptive mode generate() returns empty list (no mempool injection)."""
    sp = SyntheticPolicy(enabled=True, granularity_mode="adaptive")
    txs = sp.generate(0, 5000)
    assert txs == []


def test_legacy_granular_still_works():
    """Granular mode generates fixed-count txs for mempool."""
    sp = SyntheticPolicy(actions_per_block=100, fee_per_action=5000,
                         granularity_mode="granular", tx_granularity_actions=1)
    txs = sp.generate(0, 5000)
    assert len(txs) == 100
    assert all(tx.logical_actions == 1 for tx in txs)
    assert sum(tx.fee_paid for tx in txs) == 100 * 5000


def test_legacy_atomic_still_works():
    """Atomic mode generates one large tx for mempool."""
    sp = SyntheticPolicy(actions_per_block=100, fee_per_action=5000,
                         granularity_mode="atomic")
    txs = sp.generate(0, 5000)
    assert len(txs) == 1
    assert txs[0].logical_actions == 100


def test_legacy_granular_remainder():
    sp = SyntheticPolicy(actions_per_block=103, fee_per_action=5000,
                         granularity_mode="granular", tx_granularity_actions=10)
    txs = sp.generate(0, 5000)
    assert len(txs) == 11
    assert txs[-1].logical_actions == 3


# -----------------------------------------------------------------------
# Integration tests: adaptive synthetic in full simulation
# -----------------------------------------------------------------------

def _base_cfg(**overrides):
    cfg = {
        "num_blocks": 100, "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {"type": "FixedZip317Controller", "marginal_fee": 5000},
        "block_builder": {"mode": "zip317_weighted_random"},
        "synthetic": {
            "enabled": True, "actions_per_block": 100, "fee_per_action": 5000,
            "granularity_mode": "adaptive", "median_tx_actions": 3,
        },
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": False},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def test_adaptive_synthetic_in_blocks_but_not_mempool():
    """Synthetic txs should appear in blocks (for oracle) but mempool should
    contain only honest txs."""
    m, chain = run_scenario(_base_cfg(), compute_baseline=False)
    # Every block should have synthetic txs
    for r in m.records:
        assert r.synthetic_actions_included > 0, \
            f"h{r.height}: no synthetic in block"
    # Mempool should never contain synthetic (they bypass mempool)
    for r in m.records:
        # mempool_size only counts honest (no attacker, no synthetic)
        # If synthetic were in mempool, mempool_size would be inflated
        assert r.mempool_size < 500, \
            f"h{r.height}: mempool_size={r.mempool_size}, possible synthetic accumulation"


def test_adaptive_synthetic_count_adapts_to_utilization():
    """Blocks with more real demand should have fewer synthetic samples."""
    # Low demand: lots of synthetic
    m_low, chain_low = run_scenario(
        _base_cfg(honest_demand={"arrival_rate": 5, "mean_actions": 3, "expiry_blocks": 40}),
        compute_baseline=False,
    )
    # High demand: less synthetic
    m_high, chain_high = run_scenario(
        _base_cfg(honest_demand={"arrival_rate": 80, "mean_actions": 3, "expiry_blocks": 40}),
        compute_baseline=False,
    )
    avg_syn_low = sum(r.synthetic_actions_included for r in m_low.records) / len(m_low.records)
    avg_syn_high = sum(r.synthetic_actions_included for r in m_high.records) / len(m_high.records)
    assert avg_syn_low > avg_syn_high, \
        f"Low demand should have more synthetic: low={avg_syn_low:.0f}, high={avg_syn_high:.0f}"


def test_adaptive_synthetic_never_displaced_by_attacker():
    """With adaptive synthetic, attacker filling the block does NOT eliminate
    synthetic from the oracle. Synthetic is computed from remaining capacity
    AFTER block building, so a full block gets k_i=0 synthetic. But partial
    blocks still get synthetic.

    Key difference from mempool model: if attacker uses 500 of 1000 action-cap
    and the byte usage is 125000 of 2000000, remaining = 1875000 bytes,
    so k_i = 1875000 / 750 = 2500 synthetic samples still participate.
    """
    cfg = _base_cfg(
        num_blocks=200,
        attacker={
            "enabled": True, "type": "BurstSpamAttacker",
            "start_height": 50, "end_height": 60,
            "actions_per_block": 500, "target_fee_multiplier": 10,
        },
    )
    m, chain = run_scenario(cfg)
    # During attack blocks (h50-59), synthetic should still be present
    # because byte usage is 500*250 = 125000, remaining = 1875000
    for r in m.records:
        if 50 <= r.height < 60:
            assert r.synthetic_actions_included > 0, \
                f"h{r.height}: adaptive synthetic should still be present during attack"


def test_adaptive_synthetic_oracle_anchoring_under_attack():
    """With adaptive synthetic and median-of-medians, a 10-block burst should
    not move the oracle because:
    1. Synthetic samples are always present in each block (computed from remaining bytes)
    2. Each block's median includes synthetic at floor fee
    3. Attacker can only influence their own blocks' medians
    4. With 10/50 blocks affected, window median stays at floor
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
    m, _ = run_scenario(cfg)
    s = m.summary()
    # Oracle should not have jumped beyond the round10 base bucket
    assert s["public_fee_bucket_final"] <= 10000, \
        f"Fee jumped to {s['public_fee_bucket_final']} despite adaptive synthetic anchoring"


def test_adaptive_vs_granular_attacker_cannot_displace_adaptive():
    """Compare adaptive vs granular under attack.

    With granular (mempool-based), attacker can displace synthetic from block.
    With adaptive (capacity-based), synthetic is computed from remaining bytes
    and cannot be displaced.
    """
    base = {
        "num_blocks": 150, "random_seed": 42,
        "chain": {"block_action_cap": 200, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {"type": "FixedZip317Controller", "marginal_fee": 5000},
        "block_builder": {"mode": "highest_fee_per_action"},
        "honest_demand": {"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {
            "enabled": True, "type": "FastLaneFlapAttacker",
            "start_height": 50, "end_height": 100,
            "actions_per_block": 90, "target_fee_multiplier": 1, "expiry_blocks": 2,
        },
    }

    # Granular: synthetic enters mempool, can be displaced
    cfg_gran = {**base, "synthetic": {
        "enabled": True, "actions_per_block": 100, "fee_per_action": 5000,
        "granularity_mode": "granular", "tx_granularity_actions": 1,
    }}
    m_gran, _ = run_scenario(cfg_gran, compute_baseline=False)

    # Adaptive: synthetic computed from remaining capacity, never displaced
    cfg_adapt = {**base, "synthetic": {
        "enabled": True, "fee_per_action": 5000,
        "granularity_mode": "adaptive", "median_tx_actions": 3,
    }}
    m_adapt, _ = run_scenario(cfg_adapt, compute_baseline=False)

    # During attack blocks, adaptive should have more synthetic than granular
    attack_syn_gran = [r.synthetic_actions_included for r in m_gran.records if 55 <= r.height < 95]
    attack_syn_adapt = [r.synthetic_actions_included for r in m_adapt.records if 55 <= r.height < 95]

    avg_gran = sum(attack_syn_gran) / len(attack_syn_gran) if attack_syn_gran else 0
    avg_adapt = sum(attack_syn_adapt) / len(attack_syn_adapt) if attack_syn_adapt else 0

    assert avg_adapt > avg_gran, \
        f"Adaptive synthetic ({avg_adapt:.0f}) should exceed granular ({avg_gran:.0f}) during attack"
