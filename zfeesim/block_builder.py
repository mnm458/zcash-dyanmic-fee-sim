"""Block builder: selects transactions from mempool into a block."""

from __future__ import annotations

import random

from .types import Tx
from .zip317 import weight_ratio


class BlockBuilder:
    def __init__(self, action_cap: int, byte_cap: int, mode: str = "highest_fee_per_action",
                 rng: random.Random | None = None) -> None:
        self.action_cap = action_cap
        self.byte_cap = byte_cap
        self.mode = mode
        self.rng = rng or random.Random()

    def select(self, mempool: list[Tx], marginal_fee: int) -> list[Tx]:
        if self.mode == "highest_fee_per_action":
            return self._select_highest_fee(mempool)
        elif self.mode == "zip317_weighted_random":
            return self._select_zip317_weighted(mempool, marginal_fee)
        elif self.mode == "fifo":
            return self._select_fifo(mempool)
        elif self.mode == "random":
            return self._select_random(mempool)
        else:
            return self._select_highest_fee(mempool)

    def _fits(self, selected: list[Tx], candidate: Tx) -> bool:
        cur_actions = sum(tx.logical_actions for tx in selected)
        cur_bytes = sum(tx.byte_size for tx in selected)
        return (cur_actions + candidate.logical_actions <= self.action_cap
                and cur_bytes + candidate.byte_size <= self.byte_cap)

    def _select_highest_fee(self, mempool: list[Tx]) -> list[Tx]:
        sorted_txs = sorted(mempool, key=lambda tx: tx.fee_per_action, reverse=True)
        selected: list[Tx] = []
        for tx in sorted_txs:
            if self._fits(selected, tx):
                selected.append(tx)
        return selected

    def _select_zip317_weighted(self, mempool: list[Tx], marginal_fee: int) -> list[Tx]:
        candidates = list(mempool)
        selected: list[Tx] = []
        while candidates:
            weights = [weight_ratio(tx, marginal_fee) for tx in candidates]
            total = sum(weights)
            if total <= 0:
                break
            r = self.rng.random() * total
            acc = 0.0
            chosen_idx = 0
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    chosen_idx = i
                    break
            chosen = candidates.pop(chosen_idx)
            if self._fits(selected, chosen):
                selected.append(chosen)
            # If it doesn't fit, it's removed from candidates anyway (ZIP-317 behavior)
        return selected

    def _select_fifo(self, mempool: list[Tx]) -> list[Tx]:
        sorted_txs = sorted(mempool, key=lambda tx: tx.created_height)
        selected: list[Tx] = []
        for tx in sorted_txs:
            if self._fits(selected, tx):
                selected.append(tx)
        return selected

    def _select_random(self, mempool: list[Tx]) -> list[Tx]:
        candidates = list(mempool)
        self.rng.shuffle(candidates)
        selected: list[Tx] = []
        for tx in candidates:
            if self._fits(selected, tx):
                selected.append(tx)
        return selected
