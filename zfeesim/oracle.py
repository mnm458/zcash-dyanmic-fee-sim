"""Fee oracle: computes dynamic fee estimates from recent blocks.

The audit specifies a median-of-medians structure:
  1. For each block i in the lookback window, collect per-action fees from
     real txs, append k_i synthetic samples at the floor fee, compute
     per-block median m_i = median(P'_i).
  2. Compute the window median M = median({m_i}).

Each block gets one vote regardless of tx/action count.

Legacy global-flat oracles are kept for comparison experiments.
"""

from __future__ import annotations

import math
import statistics

from .types import Block, Tx, TxKind
from .zip317 import WEIGHT_RATIO_CAP, conventional_fee

LOOKBACK: int = 50
REORG_BUFFER: int = 5

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
    """Weighted median over (value, weight) pairs."""
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


# -----------------------------------------------------------------------
# Per-block median helpers
# -----------------------------------------------------------------------

def _block_median(block: Block, include_synthetic: bool = True) -> float | None:
    """Unweighted median of fee_per_action for a single block (audit spec).

    The audit treats each tx as one sample (unweighted), not action-weighted.
    """
    fees: list[float] = []
    for tx in block.txs:
        if not _should_include(tx, include_synthetic):
            continue
        fees.append(tx.fee_per_action)
    if not fees:
        return None
    return statistics.median(fees)


def _block_action_weighted_median(block: Block,
                                  include_synthetic: bool = True) -> float | None:
    samples: list[tuple[float, int]] = []
    for tx in block.txs:
        if not _should_include(tx, include_synthetic):
            continue
        samples.append((tx.fee_per_action, tx.logical_actions))
    if not samples:
        return None
    return weighted_median(samples)


# -----------------------------------------------------------------------
# Median-of-medians oracles (audit-spec-faithful)
# -----------------------------------------------------------------------

def median_of_medians_fee(blocks: list[Block],
                          include_synthetic: bool = True) -> float:
    """Audit-spec oracle: per-block unweighted median, then median of block medians."""
    block_medians: list[float] = []
    for block in blocks:
        m = _block_median(block, include_synthetic)
        if m is not None:
            block_medians.append(m)
    if not block_medians:
        return 0.0
    return statistics.median(block_medians)


# -----------------------------------------------------------------------
# Legacy global-flat oracles (kept for comparison)
# -----------------------------------------------------------------------

def transaction_weighted_median_fee(blocks: list[Block],
                                    include_synthetic: bool = True) -> float:
    """Global flat: pool all txs, weight=1 per tx."""
    samples: list[tuple[float, int]] = []
    for block in blocks:
        for tx in block.txs:
            if not _should_include(tx, include_synthetic):
                continue
            samples.append((tx.fee_per_action, 1))
    return weighted_median(samples)


def action_weighted_median_fee(blocks: list[Block],
                               include_synthetic: bool = True) -> float:
    """Global flat: pool all txs, weight=logical_actions per tx."""
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
    """Global flat: pool all txs, action-weighted, with fee capping."""
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
    """Global flat: pool all txs, weight=byte_size per tx."""
    samples: list[tuple[float, int]] = []
    for block in blocks:
        for tx in block.txs:
            if not _should_include(tx, include_synthetic):
                continue
            samples.append((tx.fee_per_action, tx.byte_size))
    return weighted_median(samples)


# -----------------------------------------------------------------------
# Oracle sample breakdown (for reporting)
# -----------------------------------------------------------------------

def oracle_sample_breakdown(blocks: list[Block],
                            include_synthetic: bool = True) -> dict[str, int]:
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


# -----------------------------------------------------------------------
# Quantization
# -----------------------------------------------------------------------

def quantize_power_of_10(raw_fee: float) -> int:
    """Floor-based: largest power of 10 not exceeding raw_fee."""
    if raw_fee <= 0:
        return 0
    exponent = math.floor(math.log10(raw_fee))
    return int(10 ** exponent)


def quantize_round10(raw_fee: float) -> int:
    """Audit/proposal Round10: boundary at 5 * 10^n.

    If raw_fee >= 5 * 10^n, rounds up to 10^(n+1). Otherwise 10^n.
    Examples: 4999->1000, 5000->10000, 32000->10000, 78000->100000.
    """
    if raw_fee <= 0:
        return 0
    n = math.floor(math.log10(raw_fee))
    if raw_fee >= 5 * (10 ** n):
        return int(10 ** (n + 1))
    return int(10 ** n)


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
    "median_of_medians": "median_of_medians",
    "transaction_weighted_median": "transaction_weighted_median",
    "action_weighted_median": "action_weighted_median",
    "capped_effective_fee_median": "capped_effective_fee_median",
    "byte_share_weighted_median": "byte_share_weighted_median",
}


def compute_oracle_fee(blocks: list[Block], oracle_type: str,
                       marginal_fee: int = 5000,
                       block_byte_cap: int = 2_000_000,
                       include_synthetic: bool = True) -> float:
    if oracle_type == "median_of_medians":
        return median_of_medians_fee(blocks, include_synthetic)
    elif oracle_type == "transaction_weighted_median":
        return transaction_weighted_median_fee(blocks, include_synthetic)
    elif oracle_type == "action_weighted_median":
        return action_weighted_median_fee(blocks, include_synthetic)
    elif oracle_type == "capped_effective_fee_median":
        return capped_effective_fee_median(blocks, marginal_fee, include_synthetic)
    elif oracle_type == "byte_share_weighted_median":
        return byte_share_weighted_median_fee(blocks, block_byte_cap, include_synthetic)
    else:
        return median_of_medians_fee(blocks, include_synthetic)
