"""Mempool: pending transaction storage with expiry and policy filtering."""

from __future__ import annotations

from .types import Tx, TxKind
from .zip317 import policy_accepts


class Mempool:
    def __init__(self) -> None:
        self._pending: dict[str, Tx] = {}

    def add(self, txs: list[Tx]) -> None:
        for tx in txs:
            if tx.tx_id not in self._pending:
                self._pending[tx.tx_id] = tx

    def expire(self, height: int) -> list[Tx]:
        expired = [tx for tx in self._pending.values() if tx.expiry_height <= height]
        for tx in expired:
            del self._pending[tx.tx_id]
        return expired

    def filter_policy(self, marginal_fee: int) -> None:
        to_remove = [
            tx_id for tx_id, tx in self._pending.items()
            if not policy_accepts(tx, marginal_fee)
        ]
        for tx_id in to_remove:
            del self._pending[tx_id]

    def remove(self, txs: list[Tx]) -> None:
        for tx in txs:
            self._pending.pop(tx.tx_id, None)

    def snapshot(self) -> list[Tx]:
        return list(self._pending.values())

    @property
    def size(self) -> int:
        return len(self._pending)

    @property
    def total_actions(self) -> int:
        return sum(tx.logical_actions for tx in self._pending.values())

    def honest_count(self) -> int:
        return sum(1 for tx in self._pending.values() if tx.kind == TxKind.HONEST)

    def attacker_count(self) -> int:
        return sum(1 for tx in self._pending.values() if tx.kind == TxKind.ATTACKER)

    def honest_actions(self) -> int:
        return sum(tx.logical_actions for tx in self._pending.values() if tx.kind == TxKind.HONEST)

    def attacker_actions(self) -> int:
        return sum(tx.logical_actions for tx in self._pending.values() if tx.kind == TxKind.ATTACKER)
