"""Synthetic demand: artificial baseline comparables.

Supports two granularity modes:
  - granular (default): emits many small txs that partially fill remaining capacity
  - atomic: emits one large tx (legacy, all-or-nothing inclusion)
"""

from __future__ import annotations

from .tx import make_tx
from .types import Tx, TxKind


class SyntheticPolicy:
    def __init__(self, *, enabled: bool = True, actions_per_block: int = 100,
                 fee_per_action: int = 5000, expiry_blocks: int = 1,
                 byte_size_per_action: int = 250,
                 granularity_mode: str = "granular",
                 tx_granularity_actions: int = 1) -> None:
        self.enabled = enabled
        self.actions_per_block = actions_per_block
        self.fee_per_action = fee_per_action
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action
        self.granularity_mode = granularity_mode
        self.tx_granularity_actions = max(1, tx_granularity_actions)

    def generate(self, height: int, current_fee: int) -> list[Tx]:
        if not self.enabled:
            return []

        if self.granularity_mode == "atomic":
            return self._generate_atomic(height)
        return self._generate_granular(height)

    def _generate_atomic(self, height: int) -> list[Tx]:
        """Legacy: one tx with all actions."""
        tx = make_tx(
            kind=TxKind.SYNTHETIC,
            created_height=height,
            expiry_height=height + self.expiry_blocks,
            logical_actions=self.actions_per_block,
            byte_size=self.actions_per_block * self.byte_size_per_action,
            fee_paid=self.fee_per_action * self.actions_per_block,
            wallet_policy="synthetic",
        )
        return [tx]

    def _generate_granular(self, height: int) -> list[Tx]:
        """Proposal-faithful: many small txs that can partially fill capacity."""
        txs: list[Tx] = []
        remaining = self.actions_per_block
        while remaining > 0:
            actions = min(self.tx_granularity_actions, remaining)
            tx = make_tx(
                kind=TxKind.SYNTHETIC,
                created_height=height,
                expiry_height=height + self.expiry_blocks,
                logical_actions=actions,
                byte_size=actions * self.byte_size_per_action,
                fee_paid=self.fee_per_action * actions,
                wallet_policy="synthetic",
            )
            txs.append(tx)
            remaining -= actions
        return txs
