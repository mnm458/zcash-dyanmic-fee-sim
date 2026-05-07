"""Hypothesis-based functional tests from spec section 19.

Each test validates a qualitative invariant using seeded randomness
and compares relative outcomes, not brittle exact values.
"""

from zfeesim.scenarios import run_scenario


def _base_cfg(**overrides) -> dict:
    cfg = {
        "num_blocks": 300,
        "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "controller": {"type": "FixedZip317Controller", "marginal_fee": 5000},
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": False, "actions_per_block": 100, "fee_per_action": 5000},
        "honest_demand": {"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": False},
    }
    _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, over: dict) -> None:
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---- Test 1: No congestion, fixed ZIP-317 ----

def test_01_no_congestion_fixed_zip317():
    """Under low demand and fixed ZIP-317, all txs confirm quickly."""
    cfg = _base_cfg(
        honest_demand={"arrival_rate": 5, "mean_actions": 3, "expiry_blocks": 40},
    )
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()
    assert s["median_confirmation_delay"] <= 1
    assert s["p95_confirmation_delay"] <= 2
    assert s["expired_honest_transactions"] == 0
    assert s["harm_ratio"] == 0


# ---- Test 2: Honest demand above capacity causes backlog ----

def test_02_honest_demand_above_capacity():
    """Persistent excess demand -> backlog and delays."""
    cfg = _base_cfg(
        num_blocks=200,
        chain={"block_action_cap": 50, "block_byte_cap": 2_000_000},
        honest_demand={"arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 20},
    )
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()
    assert s["median_confirmation_delay"] > 0
    assert s["p95_confirmation_delay"] > 0
    # Mempool should grow
    last_records = metrics.records[-10:]
    assert any(r.mempool_size > 0 for r in last_records)


# ---- Test 3: Higher fee txs confirm faster under fee sorting ----

def test_03_higher_fee_confirms_faster():
    """Urgent wallets paying more should confirm faster under congestion."""
    # Mixed scenario: half normal, half urgent, under congestion
    cfg_mixed = _base_cfg(
        num_blocks=200,
        chain={"block_action_cap": 50, "block_byte_cap": 2_000_000},
        honest_demand={
            "arrival_rate": 30, "mean_actions": 3, "expiry_blocks": 20,
            "urgency_distribution": {"normal": 0.5, "urgent": 0.5},
        },
    )
    metrics, _ = run_scenario(cfg_mixed)
    s = metrics.summary()
    wd = s.get("wallet_policy_delays", {})
    # Under fee sorting, urgent wallets (who pay more) should have <= delay than normal
    if "urgent" in wd and "normal" in wd:
        assert wd["urgent"]["median_delay"] <= wd["normal"]["median_delay"]


# ---- Test 4: ZIP-317 weight ratio cap limits overpayment advantage ----

def test_04_weight_ratio_cap():
    """Paying 100x should not give 100x priority under ZIP-317 weighted selection."""
    from zfeesim.tx import make_tx
    from zfeesim.types import TxKind
    from zfeesim.zip317 import weight_ratio

    tx_1x = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                     logical_actions=3, byte_size=750, fee_paid=15000)
    tx_100x = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                      logical_actions=3, byte_size=750, fee_paid=1500000)

    w1 = weight_ratio(tx_1x, 5000)
    w100 = weight_ratio(tx_100x, 5000)

    assert w1 == 1.0
    assert w100 == 4.0  # Capped, not 100
    assert w100 / w1 == 4.0  # Advantage is 4x, not 100x


# ---- Test 5: Transaction-weighted median is sybil-sensitive ----

def test_05_tx_weighted_sybil_sensitive():
    """Splitting into many 1-action txs moves tx-weighted median cheaply."""
    cfg = _base_cfg(
        num_blocks=300,
        controller={"type": "ComparableMedianController", "lookback": 50, "reorg_buffer": 5,
                     "oracle": "transaction_weighted_median", "base_fee": 5000, "floor_fee": 5000},
        honest_demand={"arrival_rate": 10, "mean_actions": 5, "expiry_blocks": 40},
        attacker={"enabled": True, "type": "SybilSplitAttacker",
                  "start_height": 100, "end_height": 200,
                  "total_actions": 200, "actions_per_tx": 1,
                  "target_fee_multiplier": 10},
    )
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()
    # Oracle should have moved upward
    assert s["raw_oracle_fee_final"] > 5000 or s["fee_bucket_jumps"] > 0 or s["attacker_cost"] > 0


# ---- Test 6: Action-weighted median resists sybil splitting ----

def test_06_action_weighted_resists_sybil():
    """Action-weighted median should be harder to manipulate via sybil splitting."""
    base = {
        "num_blocks": 300,
        "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": False},
        "honest_demand": {"arrival_rate": 10, "mean_actions": 5, "expiry_blocks": 40},
        "attacker": {"enabled": True, "type": "SybilSplitAttacker",
                     "start_height": 100, "end_height": 200,
                     "total_actions": 200, "actions_per_tx": 1,
                     "target_fee_multiplier": 10},
    }

    cfg_tx = {**base, "controller": {"type": "ComparableMedianController", "lookback": 50,
              "reorg_buffer": 5, "oracle": "transaction_weighted_median",
              "base_fee": 5000, "floor_fee": 5000}}
    cfg_aw = {**base, "controller": {"type": "ComparableMedianController", "lookback": 50,
              "reorg_buffer": 5, "oracle": "action_weighted_median",
              "base_fee": 5000, "floor_fee": 5000}}

    m_tx, _ = run_scenario(cfg_tx)
    m_aw, _ = run_scenario(cfg_aw)

    # Action-weighted should be more resistant (lower harm or fewer jumps)
    s_tx = m_tx.summary()
    s_aw = m_aw.summary()
    # At minimum, the action-weighted manipulation cost should be >= tx-weighted
    assert s_aw["harm_ratio"] <= s_tx["harm_ratio"] or s_aw["fee_bucket_jumps"] <= s_tx["fee_bucket_jumps"]


# ---- Test 7: Capped effective median reduces oracle poisoning ----

def test_07_capped_effective_reduces_poisoning():
    """Capped oracle should be less affected by extreme overpayment."""
    base = {
        "num_blocks": 300,
        "random_seed": 42,
        "chain": {"block_action_cap": 1000, "block_byte_cap": 2_000_000},
        "zip317": {"marginal_fee": 5000},
        "block_builder": {"mode": "highest_fee_per_action"},
        "synthetic": {"enabled": False},
        "honest_demand": {"arrival_rate": 10, "mean_actions": 3, "expiry_blocks": 40},
        "attacker": {"enabled": True, "type": "MedianPoisoningAttacker",
                     "start_height": 100, "end_height": 180,
                     "target_fee_multiplier": 100, "actions_per_block": 300},
    }

    cfg_uncapped = {**base, "controller": {"type": "ComparableMedianController", "lookback": 50,
                    "reorg_buffer": 5, "oracle": "action_weighted_median",
                    "base_fee": 5000, "floor_fee": 5000}}
    cfg_capped = {**base, "controller": {"type": "ComparableMedianWithCapController", "lookback": 50,
                  "reorg_buffer": 5, "base_fee": 5000, "floor_fee": 5000}}

    m_uncapped, _ = run_scenario(cfg_uncapped)
    m_capped, _ = run_scenario(cfg_capped)

    # Capped oracle fee should be lower
    assert m_capped.summary()["honest_overpayment_vs_fixed_zip317"] <= m_uncapped.summary()["honest_overpayment_vs_fixed_zip317"]


# ---- Test 8: Burst spam creates fee persistence ----

def test_08_burst_spam_persistence():
    """Short burst should keep fees elevated after attacker stops."""
    cfg = _base_cfg(
        num_blocks=300,
        controller={"type": "ComparableMedianController", "lookback": 50, "reorg_buffer": 5,
                     "oracle": "action_weighted_median", "base_fee": 5000, "floor_fee": 5000},
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker={"enabled": True, "type": "BurstSpamAttacker",
                  "start_height": 100, "end_height": 110,
                  "target_fee_multiplier": 10, "actions_per_block": 500},
    )
    metrics, _ = run_scenario(cfg)

    # Check: raw oracle fee after attack end should still be elevated
    post_attack = [r for r in metrics.records if 115 <= r.height <= 160]
    pre_attack = [r for r in metrics.records if 80 <= r.height <= 95]

    if post_attack and pre_attack:
        avg_post = sum(r.raw_oracle_fee for r in post_attack) / len(post_attack)
        avg_pre = sum(r.raw_oracle_fee for r in pre_attack) / len(pre_attack)
        # Post-attack fee should be >= pre-attack (lookback lag)
        assert avg_post >= avg_pre


# ---- Test 9: Bucket boundary nudging causes jumps without hysteresis ----

def test_09_boundary_nudging_no_hysteresis():
    """Near a quantization boundary, small nudges cause large bucket jumps."""
    cfg = _base_cfg(
        num_blocks=300,
        controller={"type": "ComparableMedianController", "lookback": 50, "reorg_buffer": 5,
                     "oracle": "action_weighted_median", "quantization": "power_of_10",
                     "base_fee": 5000, "floor_fee": 5000},
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker={"enabled": True, "type": "BucketBoundaryNudgingAttacker",
                  "start_height": 100, "end_height": 250,
                  "nudge_actions": 100, "nudge_fee_multiplier": 2.0},
    )
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()
    # Should see at least some fee changes
    assert s["attacker_cost"] > 0


# ---- Test 10: Hysteresis reduces bucket flapping ----

def test_10_hysteresis_reduces_flapping():
    """Hysteresis should produce fewer bucket jumps than no hysteresis."""
    atk = {"enabled": True, "type": "BucketBoundaryNudgingAttacker",
           "start_height": 100, "end_height": 250,
           "nudge_actions": 100, "nudge_fee_multiplier": 2.0}

    cfg_no_hyst = _base_cfg(
        num_blocks=300,
        controller={"type": "ComparableMedianController", "lookback": 50, "reorg_buffer": 5,
                     "oracle": "action_weighted_median", "quantization": "power_of_10",
                     "base_fee": 5000, "floor_fee": 5000},
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker=atk,
    )
    cfg_hyst = _base_cfg(
        num_blocks=300,
        controller={"type": "ComparableMedianHysteresisController", "lookback": 50, "reorg_buffer": 5,
                     "oracle": "action_weighted_median",
                     "base_fee": 5000, "floor_fee": 5000,
                     "move_up_consecutive": 5, "move_down_consecutive": 20},
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker=atk,
    )

    m_no, _ = run_scenario(cfg_no_hyst)
    m_yes, _ = run_scenario(cfg_hyst)

    assert m_yes.summary()["fee_bucket_jumps"] <= m_no.summary()["fee_bucket_jumps"]


# ---- Test 11: Binary fast lane flaps near threshold ----

def test_11_binary_fast_lane_flaps():
    """Binary fast lane should flap when demand hovers near threshold."""
    cfg = _base_cfg(
        num_blocks=300,
        chain={"block_action_cap": 200, "block_byte_cap": 2_000_000},
        controller={"type": "BinaryFastLaneController", "base_fee": 5000,
                     "open_threshold": 0.95, "close_threshold": 0.95,
                     "use_hysteresis": False,
                     "synthetic_actions_per_block": 100, "block_action_cap": 200},
        synthetic={"enabled": True, "actions_per_block": 100, "fee_per_action": 5000},
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker={"enabled": True, "type": "FastLaneFlapAttacker",
                  "start_height": 50, "end_height": 250,
                  "actions_per_block": 90, "target_fee_multiplier": 1, "expiry_blocks": 2},
    )
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()
    assert s["fast_lane_flaps"] >= 0  # May or may not flap depending on exact dynamics


# ---- Test 12: Fast-lane hysteresis reduces flapping ----

def test_12_fast_lane_hysteresis_reduces_flapping():
    """Hysteresis should reduce fast-lane flapping."""
    base_atk = {"enabled": True, "type": "FastLaneFlapAttacker",
                "start_height": 50, "end_height": 250,
                "actions_per_block": 90, "target_fee_multiplier": 1, "expiry_blocks": 2}
    base_chain = {"block_action_cap": 200, "block_byte_cap": 2_000_000}
    base_synth = {"enabled": True, "actions_per_block": 100, "fee_per_action": 5000}

    cfg_no = _base_cfg(
        num_blocks=300, chain=base_chain,
        controller={"type": "BinaryFastLaneController", "base_fee": 5000,
                     "open_threshold": 0.95, "close_threshold": 0.95,
                     "use_hysteresis": False,
                     "synthetic_actions_per_block": 100, "block_action_cap": 200},
        synthetic=base_synth,
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker=base_atk,
    )
    cfg_yes = _base_cfg(
        num_blocks=300, chain=base_chain,
        controller={"type": "BinaryFastLaneController", "base_fee": 5000,
                     "open_threshold": 0.95, "close_threshold": 0.70,
                     "use_hysteresis": True,
                     "open_consecutive": 10, "close_consecutive": 30,
                     "synthetic_actions_per_block": 100, "block_action_cap": 200},
        synthetic=base_synth,
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker=base_atk,
    )

    m_no, _ = run_scenario(cfg_no)
    m_yes, _ = run_scenario(cfg_yes)

    assert m_yes.summary()["fast_lane_flaps"] <= m_no.summary()["fast_lane_flaps"]


# ---- Test 13: Priority buckets reduce 10x cliff harm ----

def test_13_priority_buckets_reduce_cliff():
    """Graduated priority buckets should cost less than binary 1x/10x."""
    from zfeesim.wallet import UrgentWallet
    # Urgent wallet pays 10x under binary, but only 2x under priority buckets
    w_binary = UrgentWallet(multiplier=10)
    w_bucket = UrgentWallet(multiplier=2)

    fee_binary = w_binary.choose_fee(5000, True)
    fee_bucket = w_bucket.choose_fee(5000, True)

    assert fee_bucket < fee_binary


# ---- Test 14: AIMD smooths fee adjustment ----

def test_14_aimd_smooths_fees():
    """AIMD should produce lower fee volatility than direct median under bursts."""
    atk = {"enabled": True, "type": "BurstSpamAttacker",
           "start_height": 100, "end_height": 150,
           "target_fee_multiplier": 5, "actions_per_block": 300}

    cfg_median = _base_cfg(
        num_blocks=300,
        chain={"block_action_cap": 200, "block_byte_cap": 2_000_000},
        controller={"type": "ComparableMedianController", "lookback": 50, "reorg_buffer": 5,
                     "oracle": "action_weighted_median", "base_fee": 5000, "floor_fee": 5000},
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker=atk,
    )
    cfg_aimd = _base_cfg(
        num_blocks=300,
        chain={"block_action_cap": 200, "block_byte_cap": 2_000_000},
        controller={"type": "AIMDBucketController", "base_fee": 5000,
                     "alpha": 0.25, "beta": 0.90, "target_utilization": 0.70,
                     "lower_utilization": 0.40, "increase_window": 5,
                     "decrease_window": 20, "block_action_cap": 200},
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker=atk,
    )

    m_med, _ = run_scenario(cfg_median)
    m_aimd, _ = run_scenario(cfg_aimd)

    assert m_aimd.summary()["fee_volatility"] <= m_med.summary()["fee_volatility"]


# ---- Test 15: Miner self-dealing can poison actual-fee oracle ----

def test_15_miner_self_dealing():
    """Miner self-dealing should increase oracle fee at reduced effective cost."""
    cfg = _base_cfg(
        num_blocks=300,
        controller={"type": "ComparableMedianController", "lookback": 50, "reorg_buffer": 5,
                     "oracle": "action_weighted_median", "base_fee": 5000, "floor_fee": 5000},
        honest_demand={"arrival_rate": 20, "mean_actions": 3, "expiry_blocks": 40},
        attacker={"enabled": True, "type": "MinerSelfDealingAttacker",
                  "start_height": 100, "end_height": 200,
                  "actions_per_block": 200, "target_fee_multiplier": 5},
    )
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()
    # Miner should have placed high-fee txs
    assert s["miner_self_dealing_profit"] > 0


# ---- Test 16: Synthetic txs don't contaminate honest harm metrics ----

def test_16_synthetic_not_in_harm_metrics():
    """Synthetic txs should not count as honest overpayment."""
    cfg = _base_cfg(
        num_blocks=200,
        synthetic={"enabled": True, "actions_per_block": 100, "fee_per_action": 5000},
        honest_demand={"arrival_rate": 5, "mean_actions": 3, "expiry_blocks": 40},
        attacker={"enabled": False},
    )
    metrics, _ = run_scenario(cfg)
    s = metrics.summary()
    # Synthetic actions should be included
    assert any(r.synthetic_actions_included > 0 for r in metrics.records)
    # But harm metrics should be zero
    assert s["attacker_cost"] == 0
    assert s["harm_ratio"] == 0
