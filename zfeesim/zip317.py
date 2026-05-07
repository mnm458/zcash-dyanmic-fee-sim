"""Simplified ZIP-317 fee logic."""

from __future__ import annotations

from .types import Tx

GRACE_ACTIONS: int = 2
WEIGHT_RATIO_CAP: float = 4.0
DEFAULT_MARGINAL_FEE: int = 5000


def conventional_fee(tx: Tx, marginal_fee: int) -> int:
    return marginal_fee * max(GRACE_ACTIONS, tx.logical_actions)


def weight_ratio(tx: Tx, marginal_fee: int) -> float:
    cf = conventional_fee(tx, marginal_fee)
    return min(max(1, tx.fee_paid) / cf, WEIGHT_RATIO_CAP)


def policy_accepts(tx: Tx, marginal_fee: int) -> bool:
    return tx.fee_paid >= conventional_fee(tx, marginal_fee)
