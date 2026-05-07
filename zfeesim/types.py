"""Core data types for the fee simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TxKind(str, Enum):
    HONEST = "honest"
    ATTACKER = "attacker"
    MINER_SELF = "miner_self"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class Tx:
    tx_id: str
    kind: TxKind
    created_height: int
    expiry_height: int
    logical_actions: int
    byte_size: int
    fee_paid: int
    wallet_policy: str = "normal"

    @property
    def fee_per_action(self) -> float:
        return self.fee_paid / max(1, self.logical_actions)


@dataclass
class Block:
    height: int
    txs: list[Tx] = field(default_factory=list)
    marginal_fee: int = 5000
    fast_lane_open: bool = False
    raw_oracle_fee: float | None = None
    public_fee_bucket: int | None = None

    @property
    def total_actions(self) -> int:
        return sum(tx.logical_actions for tx in self.txs)

    @property
    def total_bytes(self) -> int:
        return sum(tx.byte_size for tx in self.txs)
