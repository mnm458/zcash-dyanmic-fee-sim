"""Transaction creation helpers."""

from __future__ import annotations

import hashlib

from .types import Tx, TxKind


_COUNTER = 0


def make_tx(
    *,
    kind: TxKind,
    created_height: int,
    expiry_height: int,
    logical_actions: int,
    byte_size: int,
    fee_paid: int,
    wallet_policy: str = "normal",
    tx_id: str | None = None,
) -> Tx:
    if tx_id is None:
        global _COUNTER
        _COUNTER += 1
        raw = f"{kind.value}:{created_height}:{_COUNTER}".encode()
        tx_id = hashlib.sha256(raw).hexdigest()[:16]
    return Tx(
        tx_id=tx_id,
        kind=kind,
        created_height=created_height,
        expiry_height=expiry_height,
        logical_actions=max(1, logical_actions),
        byte_size=max(1, byte_size),
        fee_paid=max(0, fee_paid),
        wallet_policy=wallet_policy,
    )


def reset_counter() -> None:
    global _COUNTER
    _COUNTER = 0
