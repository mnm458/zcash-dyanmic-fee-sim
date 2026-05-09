"""Synthetic demand: statistical baseline comparables.

The audit specifies that synthetic samples are a statistical construct,
not real transactions competing in block building:

  For each block i, compute k_i = floor((S_max - S_used) / s_tilde_i)
  where S_max is the block byte cap, S_used is bytes used by real txs,
  and s_tilde_i is the median tx size. Append k_i synthetic samples at
  the floor fee to the per-block fee sample set.

Synthetic samples never enter the mempool, never compete for block space,
and never get displaced by real demand. They always participate in the
oracle computation for their block.

The default mode is 'adaptive' (audit-faithful). Legacy modes 'granular'
and 'atomic' are kept for comparison experiments.
"""

from __future__ import annotations

import statistics

from .tx import make_tx
from .types import Block, Tx, TxKind


class SyntheticPolicy:
    def __init__(self, *, enabled: bool = True, actions_per_block: int = 100,
                 fee_per_action: int = 5000, expiry_blocks: int = 1,
                 byte_size_per_action: int = 250,
                 granularity_mode: str = "adaptive",
                 tx_granularity_actions: int = 1,
                 block_byte_cap: int = 2_000_000,
                 median_tx_actions: int = 3) -> None:
        self.enabled = enabled
        self.actions_per_block = actions_per_block
        self.fee_per_action = fee_per_action
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action
        self.granularity_mode = granularity_mode
        self.tx_granularity_actions = max(1, tx_granularity_actions)
        self.block_byte_cap = block_byte_cap
        self.median_tx_actions = max(1, median_tx_actions)

    def generate(self, height: int, current_fee: int) -> list[Tx]:
        """Generate synthetic txs for mempool-based modes (legacy)."""
        if not self.enabled:
            return []
        if self.granularity_mode == "adaptive":
            # Adaptive mode does not generate into mempool.
            # Use generate_for_block() after block building instead.
            return []
        if self.granularity_mode == "atomic":
            return self._generate_atomic(height)
        return self._generate_granular(height)

    def generate_for_block(self, height: int, block: Block) -> list[Tx]:
        """Audit-spec adaptive synthetic: compute from remaining block capacity.

        Called AFTER block building. Synthetic samples are appended to the
        block's tx list for oracle computation only. They never entered the
        mempool and were never displaced.

        k_i = floor((S_max - S_used) / median_tx_bytes)
        """
        if not self.enabled:
            return []
        if self.granularity_mode != "adaptive":
            # Non-adaptive modes generate into mempool via generate()
            return []

        used_bytes = sum(tx.byte_size for tx in block.txs)
        remaining = max(0, self.block_byte_cap - used_bytes)
        median_tx_bytes = self.byte_size_per_action * self.median_tx_actions
        if median_tx_bytes <= 0:
            return []
        k_i = remaining // median_tx_bytes
        if k_i <= 0:
            return []

        txs: list[Tx] = []
        for _ in range(k_i):
            tx = make_tx(
                kind=TxKind.SYNTHETIC,
                created_height=height,
                expiry_height=height + self.expiry_blocks,
                logical_actions=self.median_tx_actions,
                byte_size=median_tx_bytes,
                fee_paid=self.fee_per_action * self.median_tx_actions,
                wallet_policy="synthetic",
            )
            txs.append(tx)
        return txs

    def _generate_atomic(self, height: int) -> list[Tx]:
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
