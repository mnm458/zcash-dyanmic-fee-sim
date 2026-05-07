"""Fee oracle: computes dynamic fee estimates from recent blocks."""

from __future__ import annotations

import math

from .types import Block, Tx, TxKind
from .zip317 import WEIGHT_RATIO_CAP, conventional_fee

LOOKBACK: int = 50
REORG_BUFFER: int = 5

# Conversion constant
ZATS_PER_ZEC: int = 100_000_000


def zats_to_zec(zats: int | float) -> float:
    return zats / ZATS_PER_ZEC


def get_lookback_blocks(chain: list[Block], lookback: int = LOOKBACK,
                        reorg_buffer: int = REORG_BUFFER) -> list[Block]:
    if len(chain) <= reorg_buffer:
        return []
    end = len(chain) - reorg_buffer
    start = max(0, end - lookback)
    return chain[start:end]


def weighted_median(values: list[tuple[float, int]]) -> float:
    if not values:
        return 0.0
    values = sorted(values, key=lambda x: x[0])
    total = sum(w for _, w in values)
    if total == 0:
        return 0.0
    acc = 0
    for value, weight in values:
        acc += weight
        if acc >= total / 2:
            return value
    return 0.0


def _should_include(tx: Tx, include_synthetic: bool) -> bool:
    if tx.kind == TxKind.SYNTHETIC:
        return include_synthetic
    return True


def transaction_weighted_median_fee(blocks: list[Block],
                                    include_synthetic: bool = True) -> float:
    samples: list[tuple[float, int]] = []
    for block in blocks:
        for tx in block.txs:
            if not _should_include(tx, include_synthetic):
                continue
            samples.append((tx.fee_per_action, 1))
    return weighted_median(samples)


def action_weighted_median_fee(blocks: list[Block],
                               include_synthetic: bool = True) -> float:
    samples: list[tuple[float, int]] = []
    for block in blocks:
        for tx in block.txs:
            if not _should_include(tx, include_synthetic):
                continue
            samples.append((tx.fee_per_action, tx.logical_actions))
    return weighted_median(samples)


def capped_effective_fee_per_action(tx: Tx, marginal_fee: int) -> float:
    cf = conventional_fee(tx, marginal_fee)
    capped_total = min(tx.fee_paid, int(cf * WEIGHT_RATIO_CAP))
    return capped_total / max(1, tx.logical_actions)


def capped_effective_fee_median(blocks: list[Block], marginal_fee: int,
                                include_synthetic: bool = True) -> float:
    samples: list[tuple[float, int]] = []
    for block in blocks:
        for tx in block.txs:
            if not _should_include(tx, include_synthetic):
                continue
            capped_fpa = capped_effective_fee_per_action(tx, marginal_fee)
            samples.append((capped_fpa, tx.logical_actions))
    return weighted_median(samples)


def byte_share_weighted_median_fee(blocks: list[Block],
                                   block_byte_cap: int = 2_000_000,
                                   include_synthetic: bool = True) -> float:
    """Weight each tx's fee_per_action by byte_size."""
    samples: list[tuple[float, int]] = []
    for block in blocks:
        for tx in block.txs:
            if not _should_include(tx, include_synthetic):
                continue
            samples.append((tx.fee_per_action, tx.byte_size))
    return weighted_median(samples)


# --- Oracle sample breakdown (for reporting) ---

def oracle_sample_breakdown(blocks: list[Block],
                            include_synthetic: bool = True) -> dict[str, int]:
    """Return action counts by TxKind for the txs that enter the oracle sample."""
    honest = 0
    attacker = 0
    synthetic = 0
    miner_self = 0
    for block in blocks:
        for tx in block.txs:
            if not _should_include(tx, include_synthetic):
                continue
            if tx.kind == TxKind.HONEST:
                honest += tx.logical_actions
            elif tx.kind == TxKind.ATTACKER:
                attacker += tx.logical_actions
            elif tx.kind == TxKind.SYNTHETIC:
                synthetic += tx.logical_actions
            elif tx.kind == TxKind.MINER_SELF:
                miner_self += tx.logical_actions
    return {
        "honest_actions_in_oracle": honest,
        "attacker_actions_in_oracle": attacker,
        "synthetic_actions_in_oracle": synthetic,
        "miner_self_actions_in_oracle": miner_self,
        "total_actions_in_oracle": honest + attacker + synthetic + miner_self,
    }


# --- Quantization ---

def quantize_power_of_10(raw_fee: float) -> int:
    if raw_fee <= 0:
        return 0
    exponent = round(math.log10(raw_fee))
    return int(10 ** exponent)


PRIORITY_MULTIPLIERS = [1, 2, 5, 10]


def quantize_priority_multipliers(raw_fee: float, base_fee: int) -> int:
    if base_fee <= 0 or raw_fee <= 0:
        return base_fee
    ratio = raw_fee / base_fee
    best = 1
    for m in PRIORITY_MULTIPLIERS:
        if ratio >= m:
            best = m
    return base_fee * best


ORACLE_REGISTRY: dict[str, str] = {
    "transaction_weighted_median": "transaction_weighted_median",
    "action_weighted_median": "action_weighted_median",
    "capped_effective_fee_median": "capped_effective_fee_median",
    "byte_share_weighted_median": "byte_share_weighted_median",
}


def compute_oracle_fee(blocks: list[Block], oracle_type: str,
                       marginal_fee: int = 5000,
                       block_byte_cap: int = 2_000_000,
                       include_synthetic: bool = True) -> float:
    if oracle_type == "transaction_weighted_median":
        return transaction_weighted_median_fee(blocks, include_synthetic)
    elif oracle_type == "action_weighted_median":
        return action_weighted_median_fee(blocks, include_synthetic)
    elif oracle_type == "capped_effective_fee_median":
        return capped_effective_fee_median(blocks, marginal_fee, include_synthetic)
    elif oracle_type == "byte_share_weighted_median":
        return byte_share_weighted_median_fee(blocks, block_byte_cap, include_synthetic)
    else:
        return action_weighted_median_fee(blocks, include_synthetic)
