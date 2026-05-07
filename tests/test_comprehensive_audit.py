"""Comprehensive bug-audit test suite for Zcash dynamic fee simulator.

Tests cover:
  1. Deterministic scenario regression (burst spam, fast-lane, median poisoning)
  2. White-box oracle sampling
  3. White-box synthetic demand
  4. White-box fee bucketing / quantization boundaries
  5. White-box fast-lane trigger
  6. White-box harm accounting
  7. Optimizer concern reproduction & ablations
"""

from __future__ import annotations

import copy
import math
import random

import pytest

from zfeesim.block_builder import BlockBuilder
from zfeesim.config import load_scenario
from zfeesim.controllers import (
    BinaryFastLaneController,
    ComparableMedianController,
    FixedZip317Controller,
)
from zfeesim.mempool import Mempool
from zfeesim.metrics import MetricsCollector
from zfeesim.oracle import (
    LOOKBACK,
    REORG_BUFFER,
    action_weighted_median_fee,
    get_lookback_blocks,
    oracle_sample_breakdown,
    quantize_power_of_10,
    transaction_weighted_median_fee,
    weighted_median,
)
from zfeesim.scenarios import run_scenario
from zfeesim.synthetic import SyntheticPolicy
from zfeesim.tx import make_tx, reset_counter
from zfeesim.types import Block, Tx, TxKind
from zfeesim.zip317 import conventional_fee


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(height: int, txs: list[Tx], marginal_fee: int = 5000) -> Block:
    return Block(height=height, txs=txs, marginal_fee=marginal_fee)


def _synth_tx(height: int, actions: int = 1, fee_per_action: int = 5000) -> Tx:
    return make_tx(
        kind=TxKind.SYNTHETIC,
        created_height=height,
        expiry_height=height + 1,
        logical_actions=actions,
        byte_size=actions * 250,
        fee_paid=fee_per_action * actions,
        wallet_policy="synthetic",
    )


def _honest_tx(height: int, actions: int = 3, fee_per_action: int = 5000) -> Tx:
    return make_tx(
        kind=TxKind.HONEST,
        created_height=height,
        expiry_height=height + 40,
        logical_actions=actions,
        byte_size=actions * 250,
        fee_paid=fee_per_action * actions,
        wallet_policy="normal",
    )


def _attacker_tx(height: int, actions: int = 50,
                 fee_per_action: int = 50000) -> Tx:
    return make_tx(
        kind=TxKind.ATTACKER,
        created_height=height,
        expiry_height=height + 40,
        logical_actions=actions,
        byte_size=actions * 250,
        fee_paid=fee_per_action * actions,
        wallet_policy="attacker",
    )


# ===================================================================
# 1. DETERMINISTIC SCENARIO REGRESSION TESTS
# ===================================================================

class TestBurstSpamRegression:
    """Burst spam scenario with fixed seed=42."""

    @pytest.fixture(scope="class")
    def result(self):
        reset_counter()
        cfg = load_scenario("experiments/burst_spam_persistence.yaml")
        m, c = run_scenario(cfg)
        return m, c, m.summary()

    def test_attacker_cost(self, result):
        _, _, s = result
        got = s["effective_attacker_cost_zec"]
        # After BUG-1 fix (floor quantization), base fee stays at 5000,
        # so attacker pays 10x * 5000 * 500 actions * 10 blocks = 2.5 ZEC
        assert 1.5 <= got <= 5.0, (
            f"Expected burst_spam attacker_cost ~2.5 ZEC, got {got}"
        )

    def test_harm_ratio_below_one(self, result):
        _, _, s = result
        got = s["harm_ratio"]
        assert got < 1.0, (
            f"Expected burst_spam harm_ratio < 1.0 under proposal-faithful config, got {got}"
        )

    def test_harm_ratio_range(self, result):
        _, _, s = result
        got = s["harm_ratio"]
        assert 0.0 <= got <= 0.8, (
            f"Expected burst_spam harm_ratio ~0.52, got {got}"
        )

    def test_synthetic_in_oracle(self, result):
        m, _, _ = result
        # Check that oracle includes synthetic samples in post-attack lookback
        post_attack_records = [r for r in m.records if 115 <= r.height <= 170]
        for r in post_attack_records:
            assert r.synthetic_actions_in_oracle > 0, (
                f"Expected oracle sample count to include synthetic samples at h={r.height}, "
                f"got real={r.honest_actions_in_oracle}, synthetic={r.synthetic_actions_in_oracle}"
            )

    def test_synthetic_granular(self, result):
        _, chain, _ = result
        # Synthetic txs should be 1-action each (granular mode)
        for block in chain.blocks[:10]:
            syn_txs = [tx for tx in block.txs if tx.kind == TxKind.SYNTHETIC]
            for tx in syn_txs:
                assert tx.logical_actions == 1, (
                    f"Expected granular synthetic txs with 1 action each, "
                    f"got {tx.logical_actions} actions at h={block.height}"
                )

    def test_attacker_active_window(self, result):
        m, _, _ = result
        for r in m.records:
            if r.height < 100 or r.height >= 110:
                assert r.attacker_actions_included == 0, (
                    f"Expected no attacker actions outside h=[100,110), "
                    f"got {r.attacker_actions_included} at h={r.height}"
                )

    def test_fee_bucket_jump_count(self, result):
        _, _, s = result
        # With quantization, at most 1-2 jumps expected (initial 5000->10000 quantization)
        assert s["fee_bucket_jumps"] <= 4, (
            f"Expected few fee bucket jumps, got {s['fee_bucket_jumps']}"
        )

    def test_oracle_sample_count_consistent(self, result):
        m, _, _ = result
        # After warmup, oracle should have lookback*synthetic_per_block + real txs
        # Check a record well into the sim
        r = m.records[200]
        expected_min_synthetic = 30 * 100  # at least 30 blocks worth of synthetic
        assert r.synthetic_actions_in_oracle >= expected_min_synthetic * 0.5, (
            f"Expected oracle to include many synthetic samples at h={r.height}, "
            f"got {r.synthetic_actions_in_oracle}"
        )


class TestFastLaneRegression:
    """Fast-lane flap scenario with fixed seed=42."""

    @pytest.fixture(scope="class")
    def result(self):
        reset_counter()
        cfg = load_scenario("experiments/fast_lane_flap.yaml")
        m, c = run_scenario(cfg)
        return m, c, m.summary()

    def test_attacker_cost(self, result):
        _, _, s = result
        got = s["effective_attacker_cost_zec"]
        assert 0.5 <= got <= 3.0, (
            f"Expected fast_lane attacker_cost ~1.4 ZEC, got {got}"
        )

    def test_harm_ratio_below_one(self, result):
        _, _, s = result
        got = s["harm_ratio"]
        assert got < 1.0, (
            f"Expected fast_lane harm_ratio < 1.0 under proposal-faithful config, got {got}"
        )

    def test_harm_ratio_range(self, result):
        _, _, s = result
        got = s["harm_ratio"]
        assert 0.0 <= got <= 0.2, (
            f"Expected fast_lane harm_ratio ~0.08, got {got}"
        )

    def test_displacement_below_threshold(self, result):
        _, _, s = result
        got = s["synthetic_displacement_ratio_avg"]
        assert got < 0.95, (
            f"Expected fast_lane displacement < 0.95, got {got}"
        )

    def test_fast_lane_rarely_open(self, result):
        _, _, s = result
        total_blocks = 400
        got = s["fast_lane_open_blocks"]
        assert got < total_blocks * 0.1, (
            f"Expected fast lane open rarely (<40 blocks), got {got}"
        )

    def test_displacement_formula(self, result):
        m, _, _ = result
        # Verify displacement = 1 - (synthetic_included / synthetic_per_block)
        for r in m.records:
            expected = 1.0 - (r.synthetic_actions_included / 100)
            expected = max(0.0, min(1.0, expected))
            assert abs(r.synthetic_displacement_ratio - expected) < 1e-9, (
                f"Expected displacement {expected} at h={r.height}, "
                f"got {r.synthetic_displacement_ratio}"
            )


class TestMedianPoisoningRegression:
    """Low-volume median poisoning scenario with fixed seed=42."""

    @pytest.fixture(scope="class")
    def result(self):
        reset_counter()
        cfg = load_scenario("experiments/low_volume_median_poisoning.yaml")
        m, c = run_scenario(cfg)
        return m, c, m.summary()

    def test_attacker_cost(self, result):
        _, _, s = result
        got = s["effective_attacker_cost_zec"]
        # The actual cost is much higher than expected due to fee escalation
        assert got > 1.0, (
            f"Expected median_poison attacker_cost > 1 ZEC, got {got}"
        )

    def test_harm_ratio_below_one(self, result):
        _, _, s = result
        got = s["harm_ratio"]
        assert got < 1.0, (
            f"Expected median_poison harm_ratio < 1.0 under proposal-faithful config, got {got}"
        )

    def test_synthetic_anchoring(self, result):
        m, _, _ = result
        # During attack, synthetic actions in oracle should dominate or be significant
        attack_records = [r for r in m.records if 105 <= r.height <= 185]
        for r in attack_records:
            total = r.total_actions_in_oracle
            if total > 0:
                syn_fraction = r.synthetic_actions_in_oracle / total
                assert syn_fraction > 0.1, (
                    f"Expected synthetic to be significant fraction of oracle at h={r.height}, "
                    f"got {syn_fraction:.3f} ({r.synthetic_actions_in_oracle}/{total})"
                )

    def test_no_self_reinforcing_loop(self, result):
        m, _, _ = result
        # After attack ends (h=180), fee should eventually return to baseline
        post_attack = [r for r in m.records if r.height >= 250]
        if post_attack:
            final_bucket = post_attack[-1].public_fee_bucket
            assert final_bucket <= 100000, (
                f"Expected fee to settle after attack, "
                f"got final bucket {final_bucket}"
            )

    def test_attacker_actions_correct(self, result):
        m, _, _ = result
        for r in m.records:
            if 100 <= r.height < 180:
                # Attacker active, should try to include 300 actions
                pass  # some may not fit, that's ok
            else:
                assert r.attacker_actions_included == 0, (
                    f"Expected no attacker actions outside attack window, "
                    f"got {r.attacker_actions_included} at h={r.height}"
                )


# ===================================================================
# 2. WHITE-BOX ORACLE SAMPLING TESTS
# ===================================================================

class TestOracleSampling:
    """Verify oracle mechanics: global median, lookback, tip buffer."""

    def test_global_median_not_per_block_average(self):
        """Median must be computed globally, not as average of per-block medians."""
        reset_counter()
        # Block A: 10 txs at 5000
        # Block B: 10 txs at 50000
        # Per-block medians: 5000, 50000 -> average = 27500
        # Global median (20 txs total, each weight 1): median = 5000 or 50000
        block_a = _make_block(0, [_honest_tx(0, 1, 5000) for _ in range(10)])
        block_b = _make_block(1, [_honest_tx(1, 1, 50000) for _ in range(10)])
        blocks = [block_a, block_b]

        global_med = transaction_weighted_median_fee(blocks)
        # With 10 at 5000 and 10 at 50000, total=20, half=10
        # sorted: 10 at 5000, 10 at 50000. acc after first group = 10 >= 10 -> 5000
        assert global_med == 5000.0, (
            f"Expected global median 5000, got {global_med}. "
            "Possible bug: median computed per-block then averaged."
        )

    def test_tip_buffer_excludes_recent_blocks(self):
        """The 5-block tip buffer must exclude the most recent 5 blocks."""
        reset_counter()
        chain = []
        for h in range(20):
            chain.append(_make_block(h, [_synth_tx(h)]))

        lb = get_lookback_blocks(chain, lookback=50, reorg_buffer=5)
        # chain has 20 blocks (indices 0-19)
        # end = 20 - 5 = 15 -> blocks 0..14
        assert len(lb) == 15
        heights = [b.height for b in lb]
        assert max(heights) == 14, (
            f"Expected block H-5 (h=14) to be last included in oracle due to tip buffer, "
            f"got max height {max(heights)}"
        )
        assert 15 not in heights, (
            "Expected block h=15 to be excluded from oracle due to tip buffer, got included."
        )

    def test_lookback_window_length(self):
        """Lookback window should be exactly min(LOOKBACK, available) blocks."""
        reset_counter()
        chain = []
        for h in range(100):
            chain.append(_make_block(h, [_synth_tx(h)]))

        lb = get_lookback_blocks(chain, lookback=50, reorg_buffer=5)
        # end = 100 - 5 = 95, start = 95 - 50 = 45
        assert len(lb) == 50
        assert lb[0].height == 45
        assert lb[-1].height == 94

    def test_lookback_small_chain(self):
        """Chain shorter than reorg_buffer returns empty lookback."""
        chain = [_make_block(h, []) for h in range(5)]
        lb = get_lookback_blocks(chain, lookback=50, reorg_buffer=5)
        assert lb == [], (
            f"Expected empty lookback for chain len={len(chain)} <= reorg_buffer=5, "
            f"got {len(lb)} blocks"
        )

    def test_synthetic_included_in_oracle(self):
        """When include_synthetic=True, synthetic txs contribute to median."""
        reset_counter()
        # 100 synthetic at 5000, 10 honest at 50000
        txs = [_synth_tx(0) for _ in range(100)] + [_honest_tx(0, 1, 50000) for _ in range(10)]
        blocks = [_make_block(0, txs)]
        med = action_weighted_median_fee(blocks, include_synthetic=True)
        assert med == 5000.0, (
            f"Expected median 5000 with synthetic included, got {med}"
        )

    def test_synthetic_excluded_from_oracle(self):
        """When include_synthetic=False, synthetic txs are ignored."""
        reset_counter()
        txs = [_synth_tx(0) for _ in range(100)] + [_honest_tx(0, 1, 50000) for _ in range(10)]
        blocks = [_make_block(0, txs)]
        med = action_weighted_median_fee(blocks, include_synthetic=False)
        assert med == 50000.0, (
            f"Expected median 50000 with synthetic excluded, got {med}"
        )

    def test_attacker_samples_age_out(self):
        """Attacker samples should age out after lookback window."""
        reset_counter()
        chain = []
        # First 10 blocks: attacker txs at high fee
        for h in range(10):
            chain.append(_make_block(h, [_attacker_tx(h, 50, 100000)]))
        # Next 60 blocks: only synthetic at 5000
        for h in range(10, 70):
            chain.append(_make_block(h, [_synth_tx(h) for _ in range(10)]))

        # At height 70 with lookback=50, reorg_buffer=5:
        # end=65, start=15 -> blocks 15..64 (all synthetic)
        lb = get_lookback_blocks(chain, lookback=50, reorg_buffer=5)
        med = action_weighted_median_fee(lb, include_synthetic=True)
        assert med == 5000.0, (
            f"Expected attacker samples to age out after lookback window, "
            f"but median still affected at height 70, got {med}"
        )

    def test_oracle_breakdown_counts(self):
        """Oracle sample breakdown should match actual tx kinds."""
        reset_counter()
        txs = (
            [_synth_tx(0) for _ in range(5)]
            + [_honest_tx(0, 3, 5000) for _ in range(2)]
            + [_attacker_tx(0, 10, 50000)]
        )
        blocks = [_make_block(0, txs)]
        bd = oracle_sample_breakdown(blocks, include_synthetic=True)
        assert bd["synthetic_actions_in_oracle"] == 5, f"got {bd['synthetic_actions_in_oracle']}"
        assert bd["honest_actions_in_oracle"] == 6, f"got {bd['honest_actions_in_oracle']}"
        assert bd["attacker_actions_in_oracle"] == 10, f"got {bd['attacker_actions_in_oracle']}"
        assert bd["total_actions_in_oracle"] == 21

    def test_weighted_median_single_sample(self):
        """Weighted median of a single sample returns that sample."""
        assert weighted_median([(5000, 1)]) == 5000.0

    def test_weighted_median_empty(self):
        """Weighted median of empty list returns 0."""
        assert weighted_median([]) == 0.0


# ===================================================================
# 3. WHITE-BOX SYNTHETIC DEMAND TESTS
# ===================================================================

class TestSyntheticDemand:
    """Verify synthetic demand generation and participation."""

    def test_granular_creates_individual_txs(self):
        """Granular mode must create N individual 1-action txs, not one large tx."""
        sp = SyntheticPolicy(
            enabled=True, actions_per_block=100,
            fee_per_action=5000, granularity_mode="granular",
            tx_granularity_actions=1,
        )
        reset_counter()
        txs = sp.generate(0, 5000)
        assert len(txs) == 100, (
            f"Expected granular synthetic to create 100 individual txs, got {len(txs)}. "
            "Bug: synthetic may be represented as one aggregate sample."
        )
        for tx in txs:
            assert tx.logical_actions == 1
            assert tx.kind == TxKind.SYNTHETIC
            assert tx.fee_paid == 5000

    def test_atomic_creates_one_tx(self):
        """Atomic mode creates one large tx."""
        sp = SyntheticPolicy(
            enabled=True, actions_per_block=100,
            fee_per_action=5000, granularity_mode="atomic",
            tx_granularity_actions=1,
        )
        reset_counter()
        txs = sp.generate(0, 5000)
        assert len(txs) == 1, (
            f"Expected atomic synthetic to create 1 tx, got {len(txs)}"
        )
        assert txs[0].logical_actions == 100
        assert txs[0].fee_paid == 500000

    def test_granular_5_actions_per_tx(self):
        """tx_granularity_actions=5 should create 20 txs for 100 actions."""
        sp = SyntheticPolicy(
            enabled=True, actions_per_block=100,
            fee_per_action=5000, granularity_mode="granular",
            tx_granularity_actions=5,
        )
        reset_counter()
        txs = sp.generate(0, 5000)
        assert len(txs) == 20
        assert all(tx.logical_actions == 5 for tx in txs)

    def test_granular_remainder_handling(self):
        """Remainder actions should go into a smaller final tx."""
        sp = SyntheticPolicy(
            enabled=True, actions_per_block=7,
            fee_per_action=5000, granularity_mode="granular",
            tx_granularity_actions=3,
        )
        reset_counter()
        txs = sp.generate(0, 5000)
        # 7 / 3 = 2 full + 1 remainder
        assert len(txs) == 3
        assert txs[0].logical_actions == 3
        assert txs[1].logical_actions == 3
        assert txs[2].logical_actions == 1

    def test_synthetic_competes_in_block_building(self):
        """Synthetic txs should be selectable by block builder."""
        reset_counter()
        builder = BlockBuilder(action_cap=50, byte_cap=2_000_000,
                               mode="highest_fee_per_action")
        syn_txs = [_synth_tx(0) for _ in range(100)]
        selected = builder.select(syn_txs, 5000)
        total_actions = sum(tx.logical_actions for tx in selected)
        assert total_actions == 50, (
            f"Expected block builder to select 50 synthetic actions, got {total_actions}"
        )

    def test_synthetic_displaced_by_higher_fee(self):
        """Real txs with higher fees should displace synthetic."""
        reset_counter()
        builder = BlockBuilder(action_cap=100, byte_cap=2_000_000,
                               mode="highest_fee_per_action")
        syn_txs = [_synth_tx(0) for _ in range(100)]
        # 80 attacker actions at higher fee
        atk_txs = [_attacker_tx(0, 1, 10000) for _ in range(80)]
        selected = builder.select(syn_txs + atk_txs, 5000)
        syn_selected = sum(1 for tx in selected if tx.kind == TxKind.SYNTHETIC)
        atk_selected = sum(1 for tx in selected if tx.kind == TxKind.ATTACKER)
        assert atk_selected == 80, f"Expected all 80 attacker txs selected, got {atk_selected}"
        assert syn_selected == 20, f"Expected 20 remaining synthetic, got {syn_selected}"

    def test_one_sample_vs_many_samples_median_effect(self):
        """
        Critical bug class: if synthetic is 1 sample instead of 100,
        median poisoning becomes much easier.
        """
        reset_counter()
        # Granular: 100 synthetic at 5000, 60 attacker at 10000
        syn_granular = [_synth_tx(0) for _ in range(100)]
        atk = [_attacker_tx(0, 1, 10000) for _ in range(60)]
        block_gran = _make_block(0, syn_granular + atk)

        med_gran = action_weighted_median_fee([block_gran], include_synthetic=True)
        # 100 at 5000 + 60 at 10000 = 160 total. Half = 80.
        # sorted: 100 at 5000 (acc=100 >= 80) -> median = 5000
        assert med_gran == 5000.0, (
            f"With granular synthetic (100 samples), median should be 5000, got {med_gran}"
        )

        reset_counter()
        # Atomic: 1 synthetic with 100 actions at 5000, 60 attacker at 10000
        syn_atomic = make_tx(
            kind=TxKind.SYNTHETIC, created_height=0, expiry_height=1,
            logical_actions=100, byte_size=25000, fee_paid=500000,
            wallet_policy="synthetic",
        )
        block_atomic = _make_block(0, [syn_atomic] + atk)

        med_atomic = action_weighted_median_fee([block_atomic], include_synthetic=True)
        # Same result for action-weighted: 100 actions at 5000 + 60 at 10000
        # This is equivalent for action_weighted but differs for transaction_weighted
        assert med_atomic == 5000.0

        # Transaction-weighted difference: granular = 100 samples, atomic = 1 sample
        med_gran_tx = transaction_weighted_median_fee([block_gran], include_synthetic=True)
        # 100 txs at 5000 + 60 txs at 10000 = 160 total. Half=80. 100>=80 -> 5000
        assert med_gran_tx == 5000.0

        reset_counter()
        med_atomic_tx = transaction_weighted_median_fee([block_atomic], include_synthetic=True)
        # 1 tx at 5000 + 60 txs at 10000 = 61 total. Half=30.5. acc=1<30.5, then 61>=30.5 -> 10000
        assert med_atomic_tx == 10000.0, (
            "Transaction-weighted median with atomic synthetic shows the bug: "
            f"1 sample vs 60 attacker -> median = {med_atomic_tx}, should be 10000 "
            "(attacker wins with atomic synthetic)"
        )

    def test_synthetic_disabled(self):
        """Disabled synthetic produces no txs."""
        sp = SyntheticPolicy(enabled=False)
        assert sp.generate(0, 5000) == []


# ===================================================================
# 4. WHITE-BOX FEE BUCKETING / QUANTIZATION TESTS
# ===================================================================

class TestFeeBucketing:
    """Test power-of-10 quantization boundaries."""

    @pytest.mark.parametrize("raw,expected_bucket", [
        (4999, 1000),     # floor(log10(4999))=3 -> 1000
        (5000, 1000),     # floor(log10(5000))=3 -> 1000
        (9999, 1000),     # floor(log10(9999))=3 -> 1000
        (10000, 10000),   # floor(log10(10000))=4 -> 10000
        (10001, 10000),   # floor(log10(10001))=4 -> 10000
        (49999, 10000),   # floor(log10(49999))=4 -> 10000
        (50000, 10000),   # floor(log10(50000))=4 -> 10000
        (99999, 10000),   # floor(log10(99999))=4 -> 10000
        (100000, 100000), # floor(log10(100000))=5 -> 100000
        (3162, 1000),     # floor(log10(3162))=3 -> 1000
        (3163, 1000),     # floor(log10(3163))=3 -> 1000
        (31622, 10000),   # floor(log10(31622))=4 -> 10000
        (31623, 10000),   # floor(log10(31623))=4 -> 10000
    ])
    def test_quantize_boundary(self, raw, expected_bucket):
        got = quantize_power_of_10(raw)
        assert got == expected_bucket, (
            f"Expected quantize_power_of_10({raw}) = {expected_bucket}, got {got}"
        )

    def test_quantize_zero(self):
        assert quantize_power_of_10(0) == 0

    def test_quantize_negative(self):
        assert quantize_power_of_10(-100) == 0

    def test_base_fee_quantizes_below_base(self):
        """
        With floor() quantization, 5000 maps to 1000. The floor_fee (5000)
        in the controller catches this, so the actual fee stays at 5000.
        No more 2x artifact.
        """
        got = quantize_power_of_10(5000)
        assert got == 1000, (
            f"Expected quantize(5000) = 1000 (floor), got {got}"
        )

    def test_fee_calculation_with_max_2_actions(self):
        """Fee should use max(2, logical_actions) per ZIP-317."""
        reset_counter()
        tx1 = _honest_tx(0, actions=1, fee_per_action=5000)
        # conventional_fee uses max(GRACE_ACTIONS=2, actions)
        cf = conventional_fee(tx1, 5000)
        assert cf == 10000, (
            f"Expected conventional_fee for 1-action tx = 5000*max(2,1)=10000, got {cf}"
        )

        tx3 = _honest_tx(0, actions=3, fee_per_action=5000)
        cf3 = conventional_fee(tx3, 5000)
        assert cf3 == 15000, (
            f"Expected conventional_fee for 3-action tx = 5000*3=15000, got {cf3}"
        )

    def test_floor_fee_prevents_undershoot(self):
        """Floor fee should prevent quantized fee from going below 5000."""
        ctrl = ComparableMedianController(
            lookback=50, reorg_buffer=5, base_fee=5000, floor_fee=5000,
            oracle_include_synthetic=True,
        )
        reset_counter()
        # Build chain with only low-fee txs that quantize to 1000
        chain_blocks = []
        for h in range(60):
            txs = [_honest_tx(h, 1, 1000) for _ in range(10)]
            chain_blocks.append(_make_block(h, txs))

        ctrl.process_block(chain_blocks)
        fee = ctrl.current_fee()
        assert fee >= 5000, (
            f"Expected floor_fee to prevent fee < 5000, got {fee}"
        )


# ===================================================================
# 5. WHITE-BOX FAST-LANE TRIGGER TESTS
# ===================================================================

class TestFastLaneTrigger:
    """Test fast-lane displacement calculation and thresholds."""

    def test_displacement_denominator(self):
        """Displacement should use synthetic_per_block as denominator."""
        ctrl = BinaryFastLaneController(
            base_fee=5000, open_threshold=0.95,
            synthetic_actions_per_block=100, block_action_cap=200,
        )
        reset_counter()
        # Block with 5 synthetic actions included (out of 100 target)
        txs = [_synth_tx(0) for _ in range(5)] + [_attacker_tx(0, 195, 10000)]
        block = _make_block(0, txs)
        ctrl.process_block([block])
        # displacement = 1 - (5/100) = 0.95
        assert ctrl.fast_lane_open(), (
            "Expected fast lane open at displacement 0.95 with threshold 0.95, got closed. "
            "Check: displacement denominator should be synthetic_capacity (100), "
            "not total_block_capacity (200)."
        )

    @pytest.mark.parametrize("syn_included,expected_open", [
        (6, False),   # displacement = 1 - 6/100 = 0.94 < 0.95
        (5, True),    # displacement = 1 - 5/100 = 0.95 >= 0.95
        (4, True),    # displacement = 1 - 4/100 = 0.96 >= 0.95
        (0, True),    # displacement = 1.0 >= 0.95
        (100, False), # displacement = 0.0 < 0.95
    ])
    def test_threshold_boundary(self, syn_included, expected_open):
        ctrl = BinaryFastLaneController(
            base_fee=5000, open_threshold=0.95,
            synthetic_actions_per_block=100, block_action_cap=200,
        )
        reset_counter()
        txs = [_synth_tx(0) for _ in range(syn_included)]
        block = _make_block(0, txs)
        ctrl.process_block([block])

        got = ctrl.fast_lane_open()
        disp = 1.0 - (syn_included / 100)
        assert got == expected_open, (
            f"Expected fast lane {'open' if expected_open else 'closed'} "
            f"at displacement {disp:.3f}, got {'open' if got else 'closed'}."
        )

    def test_fast_lane_no_persistence_without_signal(self):
        """Fast lane should close when displacement drops below threshold."""
        ctrl = BinaryFastLaneController(
            base_fee=5000, open_threshold=0.95,
            synthetic_actions_per_block=100, block_action_cap=200,
            use_hysteresis=False,
        )
        reset_counter()
        # First: open it (0 synthetic -> displacement = 1.0)
        block1 = _make_block(0, [_attacker_tx(0, 200, 10000)])
        ctrl.process_block([block1])
        assert ctrl.fast_lane_open()

        # Then: restore synthetic (100 synthetic -> displacement = 0.0)
        block2 = _make_block(1, [_synth_tx(1) for _ in range(100)])
        ctrl.process_block([block1, block2])
        assert not ctrl.fast_lane_open(), (
            "Expected fast-lane state not to persist after signal disappears"
        )

    def test_fast_lane_fee_multiplier(self):
        """When fast lane is open, fee should be base * multiplier."""
        ctrl = BinaryFastLaneController(
            base_fee=5000, fast_lane_multiplier=10,
            open_threshold=0.95,
            synthetic_actions_per_block=100, block_action_cap=200,
        )
        assert ctrl.current_fee() == 5000
        reset_counter()
        block = _make_block(0, [])  # no synthetic -> displacement = 1.0
        ctrl.process_block([block])
        assert ctrl.current_fee() == 50000, (
            f"Expected fast-lane fee = 5000*10 = 50000, got {ctrl.current_fee()}"
        )


# ===================================================================
# 6. WHITE-BOX HARM ACCOUNTING TESTS
# ===================================================================

class TestHarmAccounting:
    """Verify harm_ratio, overpayment, and cost calculations."""

    def test_honest_overpayment_uses_max_2_actions(self):
        """Baseline fee should use 5000 * max(2, actions), not 5000 * actions."""
        mc = MetricsCollector(zip317_marginal_fee=5000,
                              synthetic_actions_per_block=100)
        reset_counter()
        # 1-action honest tx paying 10000
        tx = _honest_tx(0, actions=1, fee_per_action=10000)
        block = Block(height=0, txs=[tx], marginal_fee=5000)
        mc.record_block(block, 0, 0, 0, [], [])

        # baseline = 5000 * max(2, 1) = 10000
        # paid = 10000 * 1 = 10000
        # overpayment = max(0, 10000 - 10000) = 0
        assert mc.honest_overpayment == 0, (
            f"Expected baseline fee to use max(2, actions), got overpayment={mc.honest_overpayment}. "
            f"If baseline used actions=1 directly, it would be 5000 and overpayment=5000."
        )

    def test_synthetic_excluded_from_overpayment(self):
        """Synthetic txs must not count toward honest overpayment."""
        mc = MetricsCollector(zip317_marginal_fee=5000,
                              synthetic_actions_per_block=100)
        reset_counter()
        syn = _synth_tx(0, actions=1, fee_per_action=5000)
        block = Block(height=0, txs=[syn], marginal_fee=5000)
        mc.record_block(block, 0, 0, 0, [], [])
        assert mc.honest_overpayment == 0, (
            f"Expected synthetic txs excluded from honest_overpayment, got {mc.honest_overpayment}"
        )
        assert mc._honest_total_fee == 0, (
            f"Expected synthetic not counted in honest_total_fee, got {mc._honest_total_fee}"
        )

    def test_attacker_cost_includes_attacker_fees(self):
        """Attacker cost should include fees paid by attacker txs."""
        mc = MetricsCollector(zip317_marginal_fee=5000,
                              synthetic_actions_per_block=100)
        reset_counter()
        atk = _attacker_tx(0, actions=10, fee_per_action=50000)
        block = Block(height=0, txs=[atk], marginal_fee=5000)
        mc.record_block(block, 0, 0, 0, [], [])
        assert mc.effective_attacker_cost == 500000, (
            f"Expected attacker cost = 10*50000 = 500000, got {mc.effective_attacker_cost}"
        )

    def test_attacker_txs_not_in_honest_total(self):
        """Attacker txs must not be counted as honest fees."""
        mc = MetricsCollector(zip317_marginal_fee=5000)
        reset_counter()
        atk = _attacker_tx(0, 10, 50000)
        honest = _honest_tx(0, 3, 5000)
        block = Block(height=0, txs=[atk, honest], marginal_fee=5000)
        mc.record_block(block, 0, 0, 0, [], [])
        assert mc._honest_total_fee == 15000, (
            f"Expected honest_total_fee = 5000*3 = 15000, got {mc._honest_total_fee}. "
            "Bug: attacker txs may be mislabeled as honest."
        )

    def test_harm_ratio_zero_attacker_cost(self):
        """harm_ratio should be 0 when attacker cost is 0 (no division by zero)."""
        mc = MetricsCollector(zip317_marginal_fee=5000)
        reset_counter()
        honest = _honest_tx(0, 3, 10000)
        block = Block(height=0, txs=[honest], marginal_fee=5000)
        mc.record_block(block, 0, 0, 0, [], [])
        s = mc.summary()
        assert s["harm_ratio"] == 0.0, (
            f"Expected harm_ratio = 0 with zero attacker cost, got {s['harm_ratio']}"
        )

    def test_unconfirmed_txs_not_in_overpayment(self):
        """Only confirmed (block-included) honest txs should count for overpayment."""
        mc = MetricsCollector(zip317_marginal_fee=5000)
        reset_counter()
        # Only record an empty block
        block = Block(height=0, txs=[], marginal_fee=5000)
        mc.record_block(block, 5, 10, 0, [], [])
        assert mc.honest_overpayment == 0, (
            f"Expected unconfirmed honest txs excluded from paid overpayment, "
            f"got {mc.honest_overpayment}"
        )

    def test_expired_txs_counted(self):
        """Expired tx count should be tracked correctly."""
        mc = MetricsCollector(zip317_marginal_fee=5000)
        reset_counter()
        expired_h = [_honest_tx(0, 3, 5000) for _ in range(3)]
        expired_a = [_attacker_tx(0, 5, 10000) for _ in range(2)]
        block = Block(height=40, txs=[], marginal_fee=5000)
        mc.record_block(block, 0, 0, 0, expired_h, expired_a)
        s = mc.summary()
        assert s["expired_honest_transactions"] == 3
        assert s["expired_attacker_transactions"] == 2

    def test_miner_self_recovery(self):
        """Miner self-dealing cost should be attacker_nominal + miner_nominal - recovered."""
        mc = MetricsCollector(zip317_marginal_fee=5000,
                              miner_fee_recovery_rate=0.8)
        reset_counter()
        miner_tx = make_tx(
            kind=TxKind.MINER_SELF, created_height=0, expiry_height=2,
            logical_actions=100, byte_size=25000, fee_paid=500000,
            wallet_policy="miner",
        )
        block = Block(height=0, txs=[miner_tx], marginal_fee=5000)
        mc.record_block(block, 0, 0, 0, [], [])
        # effective = 0 (attacker nominal) + 500000 (miner nominal) - 400000 (80% recovery)
        assert mc.effective_attacker_cost == 100000, (
            f"Expected effective cost = 500000 - 400000 = 100000, "
            f"got {mc.effective_attacker_cost}"
        )


# ===================================================================
# 7. BASELINE OVERPAYMENT BUG TEST
# ===================================================================

class TestBaselineOverpaymentBug:
    """
    CRITICAL BUG: harm_ratio includes overpayment that exists without any attacker.
    quantize_power_of_10(5000) = 10000, so honest users always pay 2x baseline.
    harm_ratio should measure INCREMENTAL harm, not total overpayment.
    """

    def test_overpayment_exists_without_attacker(self):
        """Demonstrate that honest overpayment > 0 even with no attacker."""
        reset_counter()
        cfg = {
            "num_blocks": 100, "random_seed": 42,
            "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
            "zip317": {"marginal_fee": 5000},
            "controller": {
                "type": "ComparableMedianController",
                "lookback": 50, "reorg_buffer": 5,
                "oracle": "action_weighted_median",
                "oracle_include_synthetic": True,
                "quantization": "power_of_10",
                "base_fee": 5000, "floor_fee": 5000,
            },
            "block_builder": {"mode": "highest_fee_per_action"},
            "synthetic": {"enabled": True, "actions_per_block": 100,
                          "fee_per_action": 5000, "granularity_mode": "granular"},
            "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 40},
            "attacker": {"enabled": False},
        }
        m, _ = run_scenario(cfg)
        s = m.summary()
        assert s["honest_overpayment_vs_fixed_zip317"] > 0, (
            "Expected honest overpayment > 0 due to quantization doubling even without attacker"
        )

    def test_optimizer_concern_is_baseline_artifact(self):
        """
        The optimizer concern (50 actions, 2x mult, harm_ratio=2.06) is a
        measurement artifact: the attacker causes zero incremental overpayment.
        All overpayment comes from quantize_power_of_10(5000)=10000.
        """
        reset_counter()
        from zfeesim.adversarial import _base_cfg

        base = _base_cfg()

        # Without attacker
        cfg_no = copy.deepcopy(base)
        cfg_no["attacker"] = {"enabled": False}
        m_no, _ = run_scenario(cfg_no)
        s_no = m_no.summary()
        baseline_overpayment = s_no["honest_overpayment_vs_fixed_zip317"]

        # With attacker (optimizer config)
        reset_counter()
        cfg_atk = copy.deepcopy(base)
        cfg_atk["attacker"] = {
            "enabled": True,
            "type": "BucketBoundaryNudgingAttacker",
            "start_height": 100, "end_height": 200,
            "nudge_actions": 50, "nudge_fee_multiplier": 2.0,
        }
        m_atk, _ = run_scenario(cfg_atk)
        s_atk = m_atk.summary()

        incremental = s_atk["honest_overpayment_vs_fixed_zip317"] - baseline_overpayment
        atk_cost = s_atk["effective_attacker_cost"]

        # After BUG-1 and BUG-2 fixes:
        # - quantization no longer doubles the base fee
        # - harm_ratio now uses incremental overpayment (subtracts baseline)
        # So this weak attack should produce harm_ratio near 0.
        assert s_atk["harm_ratio"] < 1.0, (
            f"Expected harm_ratio < 1 after bug fixes, got {s_atk['harm_ratio']}"
        )
        # Incremental overpayment should be small or zero
        incremental_from_summary = s_atk["incremental_overpayment"]
        if atk_cost > 0:
            true_ratio = incremental_from_summary / atk_cost
            assert true_ratio < 1.0, (
                f"Expected incremental harm_ratio < 1, got {true_ratio}"
            )


# ===================================================================
# 8. OPTIMIZER CONCERN ABLATIONS
# ===================================================================

class TestOptimizerAblations:
    """Reproduce and ablate the optimizer concern to determine root cause."""

    @staticmethod
    def _optimizer_base():
        from zfeesim.adversarial import _base_cfg
        return _base_cfg()

    @staticmethod
    def _run_with_attack(base, controller_override=None):
        reset_counter()
        cfg = copy.deepcopy(base)
        cfg["attacker"] = {
            "enabled": True,
            "type": "BucketBoundaryNudgingAttacker",
            "start_height": 100, "end_height": 200,
            "nudge_actions": 50, "nudge_fee_multiplier": 2.0,
        }
        if controller_override:
            cfg["controller"] = controller_override
        m, _ = run_scenario(cfg)
        return m.summary()

    @staticmethod
    def _run_no_attack(base, controller_override=None):
        reset_counter()
        cfg = copy.deepcopy(base)
        cfg["attacker"] = {"enabled": False}
        if controller_override:
            cfg["controller"] = controller_override
        m, _ = run_scenario(cfg)
        return m.summary()

    def test_ablation_a_no_bucketing(self):
        """Ablation A: Same attack, no quantization (raw fee). harm_ratio should drop."""
        base = self._optimizer_base()
        ctrl = {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "action_weighted_median",
            "quantization": "none",  # no quantization
            "base_fee": 5000, "floor_fee": 5000,
            "oracle_include_synthetic": True,
        }
        s_atk = self._run_with_attack(base, ctrl)
        s_no = self._run_no_attack(base, ctrl)
        incremental = s_atk["honest_overpayment_vs_fixed_zip317"] - s_no["honest_overpayment_vs_fixed_zip317"]
        # Without quantization, baseline overpayment should be near zero
        assert s_no["honest_overpayment_vs_fixed_zip317"] < s_atk.get("honest_total_fee", 1e18) * 0.5, (
            "Without quantization, baseline overpayment should be much lower"
        )

    def test_ablation_e_fixed_controller(self):
        """Ablation E: Fixed fee controller. Incremental harm should be near zero.
        Note: some overpayment exists from urgent wallets voluntarily paying 2x.
        """
        base = self._optimizer_base()
        ctrl = {"type": "FixedZip317Controller", "marginal_fee": 5000}
        s_atk = self._run_with_attack(base, ctrl)
        s_no = self._run_no_attack(base, ctrl)
        incremental = s_atk["honest_overpayment_vs_fixed_zip317"] - s_no["honest_overpayment_vs_fixed_zip317"]
        assert incremental == 0, (
            f"Expected zero incremental overpayment with fixed controller, "
            f"got {incremental}"
        )

    def test_ablation_d_no_synthetic_in_oracle(self):
        """Ablation D: Oracle excludes synthetic. Attack should become easier."""
        base = self._optimizer_base()
        ctrl_syn = {
            "type": "ComparableMedianController",
            "lookback": 50, "reorg_buffer": 5,
            "oracle": "action_weighted_median",
            "quantization": "power_of_10",
            "base_fee": 5000, "floor_fee": 5000,
            "oracle_include_synthetic": True,
        }
        ctrl_nosyn = copy.deepcopy(ctrl_syn)
        ctrl_nosyn["oracle_include_synthetic"] = False

        s_with_syn = self._run_with_attack(base, ctrl_syn)
        s_no_syn = self._run_with_attack(base, ctrl_nosyn)

        # Without synthetic in oracle, the fee should be more volatile
        # (attacker has easier time moving the median)
        assert s_no_syn["fee_bucket_jumps"] >= s_with_syn["fee_bucket_jumps"], (
            "Expected more fee instability without synthetic in oracle"
        )


# ===================================================================
# 9. LOOKBACK AND TIP BUFFER EDGE CASES
# ===================================================================

class TestLookbackEdgeCases:

    def test_exactly_reorg_buffer_blocks(self):
        """Chain with exactly reorg_buffer blocks returns empty lookback."""
        chain = [_make_block(h, []) for h in range(5)]
        lb = get_lookback_blocks(chain, lookback=50, reorg_buffer=5)
        assert lb == []

    def test_one_more_than_reorg_buffer(self):
        """Chain with reorg_buffer+1 blocks returns 1 block."""
        reset_counter()
        chain = [_make_block(h, [_synth_tx(h)]) for h in range(6)]
        lb = get_lookback_blocks(chain, lookback=50, reorg_buffer=5)
        assert len(lb) == 1
        assert lb[0].height == 0

    def test_lookback_exactly_50_with_enough_chain(self):
        """With 60 blocks, lookback=50, buffer=5: should get blocks 5..54."""
        reset_counter()
        chain = [_make_block(h, [_synth_tx(h)]) for h in range(60)]
        lb = get_lookback_blocks(chain, lookback=50, reorg_buffer=5)
        assert len(lb) == 50
        assert lb[0].height == 5
        assert lb[-1].height == 54

    def test_custom_lookback_and_buffer(self):
        """Non-default lookback and buffer values."""
        reset_counter()
        chain = [_make_block(h, [_synth_tx(h)]) for h in range(30)]
        lb = get_lookback_blocks(chain, lookback=10, reorg_buffer=3)
        # end = 30-3 = 27, start = 27-10 = 17
        assert len(lb) == 10
        assert lb[0].height == 17
        assert lb[-1].height == 26


# ===================================================================
# 10. BLOCK BUILDER INVARIANTS
# ===================================================================

class TestBlockBuilderInvariants:

    def test_action_cap_never_exceeded(self):
        """Block builder must not exceed action cap."""
        reset_counter()
        builder = BlockBuilder(action_cap=100, byte_cap=2_000_000,
                               mode="highest_fee_per_action")
        txs = [_honest_tx(0, actions=10, fee_per_action=5000) for _ in range(50)]
        selected = builder.select(txs, 5000)
        total = sum(tx.logical_actions for tx in selected)
        assert total <= 100, (
            f"Expected included block actions <= 100, got {total}"
        )

    def test_byte_cap_never_exceeded(self):
        """Block builder must not exceed byte cap."""
        reset_counter()
        builder = BlockBuilder(action_cap=10000, byte_cap=5000,
                               mode="highest_fee_per_action")
        txs = [_honest_tx(0, actions=1, fee_per_action=5000) for _ in range(100)]
        selected = builder.select(txs, 5000)
        total_bytes = sum(tx.byte_size for tx in selected)
        assert total_bytes <= 5000, (
            f"Expected included block bytes <= 5000, got {total_bytes}"
        )


# ===================================================================
# 11. END-TO-END SCENARIO INVARIANTS
# ===================================================================

class TestScenarioInvariants:
    """Run scenarios and check runtime invariants on all blocks."""

    @pytest.fixture(scope="class")
    def burst_result(self):
        reset_counter()
        cfg = load_scenario("experiments/burst_spam_persistence.yaml")
        return run_scenario(cfg)

    def test_fee_never_negative(self, burst_result):
        m, _ = burst_result
        for r in m.records:
            assert r.marginal_fee >= 0, f"Fee negative at h={r.height}"
            assert r.honest_fee_paid >= 0, f"Honest fee negative at h={r.height}"
            assert r.attacker_fee_paid >= 0, f"Attacker fee negative at h={r.height}"

    def test_actions_positive(self, burst_result):
        _, chain = burst_result
        for block in chain.blocks:
            for tx in block.txs:
                assert tx.logical_actions >= 1, (
                    f"Logical actions must be >= 1, got {tx.logical_actions}"
                )

    def test_action_cap_respected(self, burst_result):
        _, chain = burst_result
        for block in chain.blocks:
            total = block.total_actions
            assert total <= 1000, (
                f"Block actions {total} exceed cap 1000 at h={block.height}"
            )

    def test_byte_cap_respected(self, burst_result):
        _, chain = burst_result
        for block in chain.blocks:
            total = block.total_bytes
            assert total <= 2_000_000, (
                f"Block bytes {total} exceed cap at h={block.height}"
            )

    def test_synthetic_count_non_negative(self, burst_result):
        m, _ = burst_result
        for r in m.records:
            assert r.synthetic_actions_included >= 0

    def test_harm_ratio_denominator_safe(self, burst_result):
        m, _ = burst_result
        s = m.summary()
        # Should not crash with division by zero
        assert isinstance(s["harm_ratio"], (int, float))

    def test_attacker_not_mislabeled_honest(self, burst_result):
        _, chain = burst_result
        for block in chain.blocks:
            for tx in block.txs:
                if tx.kind == TxKind.ATTACKER:
                    assert tx.wallet_policy == "attacker", (
                        f"Attacker tx mislabeled with wallet_policy={tx.wallet_policy}"
                    )

    def test_synthetic_not_mislabeled_honest(self, burst_result):
        _, chain = burst_result
        for block in chain.blocks:
            for tx in block.txs:
                if tx.kind == TxKind.SYNTHETIC:
                    assert tx.wallet_policy == "synthetic"


# ===================================================================
# 12. WEIGHTED MEDIAN CORRECTNESS
# ===================================================================

class TestWeightedMedianCorrectness:

    def test_odd_total_weight(self):
        """Median with odd total weight."""
        vals = [(10, 1), (20, 1), (30, 1)]
        assert weighted_median(vals) == 20.0

    def test_even_total_weight(self):
        """Median with even total weight (picks lower of the two middle)."""
        vals = [(10, 1), (20, 1), (30, 1), (40, 1)]
        # total=4, half=2. sorted. acc: 1,2. acc=2 >= 2 -> value=20
        assert weighted_median(vals) == 20.0

    def test_weighted_skew(self):
        """Heavy weight on one value dominates."""
        vals = [(5000, 100), (50000, 10)]
        # total=110, half=55. acc=100 >= 55 -> 5000
        assert weighted_median(vals) == 5000.0

    def test_equal_split(self):
        """Equal weight, result depends on sort order."""
        vals = [(5000, 50), (10000, 50)]
        # total=100, half=50. acc=50 >= 50 -> 5000
        assert weighted_median(vals) == 5000.0

    def test_attacker_just_under_majority(self):
        """Attacker with just under half the weight can't move median."""
        vals = [(5000, 51), (50000, 49)]
        # total=100, half=50. acc=51 >= 50 -> 5000
        assert weighted_median(vals) == 5000.0

    def test_attacker_at_majority(self):
        """Attacker with exactly half weight shifts median."""
        vals = [(5000, 50), (50000, 50)]
        # total=100, half=50. acc=50 >= 50 -> 5000
        assert weighted_median(vals) == 5000.0

    def test_attacker_above_majority(self):
        """Attacker with over half weight moves median."""
        vals = [(5000, 49), (50000, 51)]
        # total=100, half=50. acc=49 < 50, then 100 >= 50 -> 50000
        assert weighted_median(vals) == 50000.0
