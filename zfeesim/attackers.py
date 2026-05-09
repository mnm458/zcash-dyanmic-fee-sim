"""Attacker strategies for adversarial demand generation."""

from __future__ import annotations

import random

from .mempool import Mempool
from .tx import make_tx
from .types import Block, Tx, TxKind


class AttackerStrategy:
    def __init__(self, *, start_height: int = 100, end_height: int = 200,
                 rng: random.Random | None = None, **kwargs) -> None:
        self.start_height = start_height
        self.end_height = end_height
        self.rng = rng or random.Random()

    def generate(self, height: int, current_fee: int,
                 chain: list[Block], mempool: Mempool) -> list[Tx]:
        if height < self.start_height or height >= self.end_height:
            return []
        return self._attack(height, current_fee, chain, mempool)

    def _attack(self, height: int, current_fee: int,
                chain: list[Block], mempool: Mempool) -> list[Tx]:
        return []


class BurstSpamAttacker(AttackerStrategy):
    def __init__(self, *, actions_per_block: int = 500,
                 target_fee_multiplier: int = 10,
                 expiry_blocks: int = 40,
                 byte_size_per_action: int = 250, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions_per_block = actions_per_block
        self.target_fee_multiplier = target_fee_multiplier
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action

    def _attack(self, height: int, current_fee: int,
                chain: list[Block], mempool: Mempool) -> list[Tx]:
        fee_per_action = current_fee * self.target_fee_multiplier
        tx = make_tx(
            kind=TxKind.ATTACKER,
            created_height=height,
            expiry_height=height + self.expiry_blocks,
            logical_actions=self.actions_per_block,
            byte_size=self.actions_per_block * self.byte_size_per_action,
            fee_paid=fee_per_action * self.actions_per_block,
            wallet_policy="attacker",
        )
        return [tx]


class MedianPoisoningAttacker(AttackerStrategy):
    def __init__(self, *, actions_per_block: int = 300,
                 target_fee_multiplier: int = 10,
                 expiry_blocks: int = 40,
                 byte_size_per_action: int = 250, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions_per_block = actions_per_block
        self.target_fee_multiplier = target_fee_multiplier
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action

    def _attack(self, height: int, current_fee: int,
                chain: list[Block], mempool: Mempool) -> list[Tx]:
        fee_per_action = current_fee * self.target_fee_multiplier
        tx = make_tx(
            kind=TxKind.ATTACKER,
            created_height=height,
            expiry_height=height + self.expiry_blocks,
            logical_actions=self.actions_per_block,
            byte_size=self.actions_per_block * self.byte_size_per_action,
            fee_paid=fee_per_action * self.actions_per_block,
            wallet_policy="attacker",
        )
        return [tx]


class BucketBoundaryNudgingAttacker(AttackerStrategy):
    def __init__(self, *, nudge_actions: int = 50,
                 nudge_fee_multiplier: float = 1.5,
                 expiry_blocks: int = 40,
                 byte_size_per_action: int = 250, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nudge_actions = nudge_actions
        self.nudge_fee_multiplier = nudge_fee_multiplier
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action

    def _attack(self, height: int, current_fee: int,
                chain: list[Block], mempool: Mempool) -> list[Tx]:
        # Submit txs at slightly above current fee to nudge median upward
        fee_per_action = int(current_fee * self.nudge_fee_multiplier)
        tx = make_tx(
            kind=TxKind.ATTACKER,
            created_height=height,
            expiry_height=height + self.expiry_blocks,
            logical_actions=self.nudge_actions,
            byte_size=self.nudge_actions * self.byte_size_per_action,
            fee_paid=fee_per_action * self.nudge_actions,
            wallet_policy="attacker",
        )
        return [tx]


class FastLaneFlapAttacker(AttackerStrategy):
    def __init__(self, *, actions_per_block: int = 90,
                 expiry_blocks: int = 2,
                 byte_size_per_action: int = 250,
                 target_fee_multiplier: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions_per_block = actions_per_block
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action
        self.target_fee_multiplier = target_fee_multiplier

    def _attack(self, height: int, current_fee: int,
                chain: list[Block], mempool: Mempool) -> list[Tx]:
        # Submit enough to displace synthetic demand near threshold
        fee_per_action = current_fee * self.target_fee_multiplier
        tx = make_tx(
            kind=TxKind.ATTACKER,
            created_height=height,
            expiry_height=height + self.expiry_blocks,
            logical_actions=self.actions_per_block,
            byte_size=self.actions_per_block * self.byte_size_per_action,
            fee_paid=fee_per_action * self.actions_per_block,
            wallet_policy="attacker",
        )
        return [tx]


class MinerSelfDealingAttacker(AttackerStrategy):
    def __init__(self, *, actions_per_block: int = 200,
                 target_fee_multiplier: int = 5,
                 fee_recovery_rate: float = 0.8,
                 expiry_blocks: int = 2,
                 byte_size_per_action: int = 250, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions_per_block = actions_per_block
        self.target_fee_multiplier = target_fee_multiplier
        self.fee_recovery_rate = fee_recovery_rate
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action

    def _attack(self, height: int, current_fee: int,
                chain: list[Block], mempool: Mempool) -> list[Tx]:
        fee_per_action = current_fee * self.target_fee_multiplier
        tx = make_tx(
            kind=TxKind.MINER_SELF,
            created_height=height,
            expiry_height=height + self.expiry_blocks,
            logical_actions=self.actions_per_block,
            byte_size=self.actions_per_block * self.byte_size_per_action,
            fee_paid=fee_per_action * self.actions_per_block,
            wallet_policy="miner",
        )
        return [tx]


class SybilSplitAttacker(AttackerStrategy):
    def __init__(self, *, total_actions: int = 300,
                 actions_per_tx: int = 1,
                 target_fee_multiplier: int = 10,
                 expiry_blocks: int = 40,
                 byte_size_per_action: int = 250, **kwargs) -> None:
        super().__init__(**kwargs)
        self.total_actions = total_actions
        self.actions_per_tx = max(1, actions_per_tx)
        self.target_fee_multiplier = target_fee_multiplier
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action

    def _attack(self, height: int, current_fee: int,
                chain: list[Block], mempool: Mempool) -> list[Tx]:
        fee_per_action = current_fee * self.target_fee_multiplier
        txs: list[Tx] = []
        remaining = self.total_actions
        while remaining > 0:
            actions = min(self.actions_per_tx, remaining)
            tx = make_tx(
                kind=TxKind.ATTACKER,
                created_height=height,
                expiry_height=height + self.expiry_blocks,
                logical_actions=actions,
                byte_size=actions * self.byte_size_per_action,
                fee_paid=fee_per_action * actions,
                wallet_policy="attacker",
            )
            txs.append(tx)
            remaining -= actions
        return txs


class FloorFeeSaturationAttacker(AttackerStrategy):
    """Audit F1: fill blocks with floor-fee txs to cause delays without fee escalation."""

    def __init__(self, *, actions_per_block: int = 900,
                 fee_per_action: int = 5000,
                 expiry_blocks: int = 40,
                 byte_size_per_action: int = 250, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actions_per_block = actions_per_block
        self.fee_per_action = fee_per_action
        self.expiry_blocks = expiry_blocks
        self.byte_size_per_action = byte_size_per_action

    def _attack(self, height: int, current_fee: int,
                chain: list[Block], mempool: Mempool) -> list[Tx]:
        tx = make_tx(
            kind=TxKind.ATTACKER,
            created_height=height,
            expiry_height=height + self.expiry_blocks,
            logical_actions=self.actions_per_block,
            byte_size=self.actions_per_block * self.byte_size_per_action,
            fee_paid=self.fee_per_action * self.actions_per_block,
            wallet_policy="attacker",
        )
        return [tx]


ATTACKER_REGISTRY: dict[str, type] = {
    "BurstSpamAttacker": BurstSpamAttacker,
    "MedianPoisoningAttacker": MedianPoisoningAttacker,
    "BucketBoundaryNudgingAttacker": BucketBoundaryNudgingAttacker,
    "FastLaneFlapAttacker": FastLaneFlapAttacker,
    "MinerSelfDealingAttacker": MinerSelfDealingAttacker,
    "SybilSplitAttacker": SybilSplitAttacker,
    "FloorFeeSaturationAttacker": FloorFeeSaturationAttacker,
}
