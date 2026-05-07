"""Honest demand generation."""

from __future__ import annotations

import math
import random

from .tx import make_tx
from .types import Tx, TxKind
from .wallet import get_wallet


class PoissonHonestDemand:
    def __init__(self, *, arrival_rate: int = 30, mean_actions: int = 3,
                 expiry_blocks: int = 40, byte_size_per_action: int = 250,
                 urgency_distribution: dict[str, float] | None = None,
                 rng: random.Random | None = None) -> None:
        self.arrival_rate = arrival_rate
        self.mean_actions = mean_actions
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action
        self.urgency_dist = urgency_distribution or {
            "patient": 0.5, "normal": 0.4, "urgent": 0.1,
        }
        self.rng = rng or random.Random()

    def generate(self, height: int, fee: int) -> list[Tx]:
        count = self._poisson(self.arrival_rate)
        txs: list[Tx] = []
        for _ in range(count):
            wallet_name = self._pick_wallet()
            wallet = get_wallet(wallet_name)
            actions = max(1, self._poisson(self.mean_actions))
            chosen_fee = wallet.choose_fee(fee, False)
            tx = make_tx(
                kind=TxKind.HONEST,
                created_height=height,
                expiry_height=height + self.expiry_blocks,
                logical_actions=actions,
                byte_size=actions * self.byte_size_per_action,
                fee_paid=chosen_fee * actions,
                wallet_policy=wallet_name,
            )
            txs.append(tx)
        return txs

    def _poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        L = math.exp(-min(lam, 700))
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.rng.random()
            if p <= L:
                return k - 1

    def _pick_wallet(self) -> str:
        r = self.rng.random()
        acc = 0.0
        for name, prob in self.urgency_dist.items():
            acc += prob
            if r < acc:
                return name
        return "normal"
