"""Scenario runner: orchestrates a full simulation from a config dict."""

from __future__ import annotations

import copy
import random
from typing import Any

from .attackers import ATTACKER_REGISTRY, AttackerStrategy
from .block_builder import BlockBuilder
from .chain import Chain
from .controllers import CONTROLLER_REGISTRY, FeeController
from .demand import PoissonHonestDemand
from .mempool import Mempool
from .metrics import MetricsCollector
from .synthetic import SyntheticPolicy
from .tx import reset_counter
from .types import Block, TxKind


def build_controller(cfg: dict[str, Any]) -> FeeController:
    ctrl_type = cfg.get("type", "FixedZip317Controller")
    cls = CONTROLLER_REGISTRY.get(ctrl_type)
    if cls is None:
        raise ValueError(f"Unknown controller: {ctrl_type}")
    import inspect
    valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in cfg.items() if k != "type" and k in valid_params}
    return cls(**kwargs)


def build_attacker(cfg: dict[str, Any], rng: random.Random) -> AttackerStrategy | None:
    if not cfg.get("enabled", False):
        return None
    atk_type = cfg.get("type", "BurstSpamAttacker")
    cls = ATTACKER_REGISTRY.get(atk_type)
    if cls is None:
        raise ValueError(f"Unknown attacker: {atk_type}")
    # All attackers use **kwargs to forward start_height/end_height to the
    # base class, so we pass everything through rather than filtering by
    # inspect.signature (which can't see through **kwargs).
    kwargs = {k: v for k, v in cfg.items() if k not in ("type", "enabled")}
    kwargs["rng"] = rng
    return cls(**kwargs)


def _check_block_invariants(block: Block, action_cap: int, byte_cap: int,
                            controller: FeeController) -> None:
    """Runtime assertions for impossible states (debug mode)."""
    for tx in block.txs:
        assert tx.fee_paid >= 0, f"fee_paid negative: {tx.fee_paid} at h={block.height}"
        assert tx.logical_actions >= 1, f"logical_actions < 1: {tx.logical_actions} at h={block.height}"
        assert tx.byte_size >= 1, f"byte_size < 1 at h={block.height}"
        if tx.kind == TxKind.ATTACKER:
            assert tx.wallet_policy == "attacker", (
                f"Attacker tx mislabeled as {tx.wallet_policy} at h={block.height}")
        if tx.kind == TxKind.SYNTHETIC:
            assert tx.wallet_policy == "synthetic", (
                f"Synthetic tx mislabeled as {tx.wallet_policy} at h={block.height}")

    total_actions = sum(tx.logical_actions for tx in block.txs)
    total_bytes = sum(tx.byte_size for tx in block.txs)
    assert total_actions <= action_cap, (
        f"Block actions {total_actions} exceed cap {action_cap} at h={block.height}")
    assert total_bytes <= byte_cap, (
        f"Block bytes {total_bytes} exceed cap {byte_cap} at h={block.height}")
    assert block.marginal_fee >= 0, f"Negative marginal fee at h={block.height}"


def run_scenario(cfg: dict[str, Any], compute_baseline: bool = True) -> tuple[MetricsCollector, Chain]:
    # If attacker is enabled, run a no-attacker shadow simulation first
    # to compute the baseline overpayment for incremental harm_ratio.
    baseline_overpayment = 0
    if compute_baseline and cfg.get("attacker", {}).get("enabled", False):
        cfg_no_atk = copy.deepcopy(cfg)
        cfg_no_atk["attacker"]["enabled"] = False
        m_base, _ = run_scenario(cfg_no_atk, compute_baseline=False)
        baseline_overpayment = m_base.honest_overpayment

    seed = cfg.get("random_seed", 42)
    rng = random.Random(seed)
    reset_counter()
    debug_invariants = cfg.get("debug_invariants", True)

    num_blocks = cfg.get("num_blocks", 500)

    chain_cfg = cfg.get("chain", {})
    action_cap = chain_cfg.get("block_action_cap", 1000)
    byte_cap = chain_cfg.get("block_byte_cap", 2_000_000)

    zip317_cfg = cfg.get("zip317", {})
    zip317_fee = zip317_cfg.get("marginal_fee", 5000)

    controller = build_controller(cfg.get("controller", {}))

    bb_cfg = cfg.get("block_builder", {})
    builder = BlockBuilder(action_cap, byte_cap, bb_cfg.get("mode", "highest_fee_per_action"), rng=rng)

    synth_cfg = cfg.get("synthetic", {})
    synthetic = SyntheticPolicy(
        enabled=synth_cfg.get("enabled", False),
        actions_per_block=synth_cfg.get("actions_per_block", 100),
        fee_per_action=synth_cfg.get("fee_per_action", zip317_fee),
        expiry_blocks=synth_cfg.get("expiry_blocks", 1),
        granularity_mode=synth_cfg.get("granularity_mode", "granular"),
        tx_granularity_actions=synth_cfg.get("tx_granularity_actions", 1),
    )

    demand_cfg = cfg.get("honest_demand", {})
    demand = PoissonHonestDemand(
        arrival_rate=demand_cfg.get("arrival_rate", 30),
        mean_actions=demand_cfg.get("mean_actions", 3),
        expiry_blocks=demand_cfg.get("expiry_blocks", 40),
        urgency_distribution=demand_cfg.get("urgency_distribution"),
        rng=rng,
    )

    attacker = build_attacker(cfg.get("attacker", {}), rng)

    atk_cfg = cfg.get("attacker", {})
    miner_recovery = atk_cfg.get("fee_recovery_rate", 0.0) if atk_cfg.get("type") == "MinerSelfDealingAttacker" else 0.0

    metrics = MetricsCollector(
        zip317_marginal_fee=zip317_fee,
        synthetic_actions_per_block=synth_cfg.get("actions_per_block", 100),
        miner_fee_recovery_rate=miner_recovery,
        baseline_overpayment=baseline_overpayment,
    )

    mempool = Mempool()
    chain = Chain()

    for h in range(num_blocks):
        current_fee = controller.current_fee()
        fast_lane = controller.fast_lane_open()

        # 1. Generate honest demand
        honest_txs = demand.generate(h, current_fee)
        mempool.add(honest_txs)

        # 2. Generate attacker demand
        attacker_txs = []
        if attacker is not None:
            attacker_txs = attacker.generate(h, current_fee, chain.blocks, mempool)
            mempool.add(attacker_txs)

        # 3. Generate synthetic demand
        synth_txs = synthetic.generate(h, current_fee)
        mempool.add(synth_txs)

        # 4. Expire old transactions
        expired = mempool.expire(h)
        expired_honest = [tx for tx in expired if tx.kind == TxKind.HONEST]
        expired_attacker = [tx for tx in expired if tx.kind == TxKind.ATTACKER]

        # 5. Block builder selects transactions
        candidates = mempool.snapshot()
        selected = builder.select(candidates, current_fee)

        # 6. Build block
        block = Block(
            height=h,
            txs=selected,
            marginal_fee=current_fee,
            fast_lane_open=fast_lane,
        )

        # 6b. Runtime invariant checks
        if debug_invariants:
            _check_block_invariants(block, action_cap, byte_cap, controller)

        # 7. Remove confirmed from mempool
        mempool.remove(selected)

        # 8. Add block to chain
        chain.append(block)

        # 9. Update controller
        controller.process_block(chain.blocks)

        # 10. Set oracle/bucket info on block
        if hasattr(controller, "raw_oracle_fee"):
            block.raw_oracle_fee = controller.raw_oracle_fee
        else:
            block.raw_oracle_fee = float(current_fee)
        block.public_fee_bucket = controller.current_fee()
        block.fast_lane_open = controller.fast_lane_open()

        # 10b. Oracle sample breakdown for reporting
        from .oracle import get_lookback_blocks, oracle_sample_breakdown
        include_syn = True
        if hasattr(controller, "include_synthetic"):
            include_syn = controller.include_synthetic
        lb = get_lookback_blocks(chain.blocks,
                                 getattr(controller, "lookback", 50),
                                 getattr(controller, "reorg_buffer", 5))
        oracle_bd = oracle_sample_breakdown(lb, include_synthetic=include_syn) if lb else {}

        # 11. Record metrics
        metrics.record_block(
            block,
            mempool_size=mempool.size,
            mempool_honest_actions=mempool.honest_actions(),
            mempool_attacker_actions=mempool.attacker_actions(),
            expired_honest=expired_honest,
            expired_attacker=expired_attacker,
            oracle_breakdown=oracle_bd,
        )

    return metrics, chain
