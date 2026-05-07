"""Tests for block builder transaction selection."""

import random

from zfeesim.block_builder import BlockBuilder
from zfeesim.tx import make_tx
from zfeesim.types import TxKind


def _tx(actions: int, fee: int) -> "Tx":
    return make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                   logical_actions=actions, byte_size=actions * 250, fee_paid=fee)


def test_highest_fee_selects_highest_first():
    bb = BlockBuilder(action_cap=10, byte_cap=100000, mode="highest_fee_per_action")
    txs = [_tx(3, 15000), _tx(3, 30000), _tx(3, 9000)]
    selected = bb.select(txs, 5000)
    # Should fit all 9 actions <= 10 cap
    assert len(selected) == 3
    # First selected should be highest fee/action
    assert selected[0].fee_paid == 30000


def test_action_cap_enforced():
    bb = BlockBuilder(action_cap=5, byte_cap=100000, mode="highest_fee_per_action")
    txs = [_tx(3, 30000), _tx(3, 15000)]
    selected = bb.select(txs, 5000)
    # Can only fit one 3-action tx (3 <= 5, but 3+3=6 > 5)
    assert len(selected) == 1
    assert selected[0].fee_paid == 30000


def test_byte_cap_enforced():
    bb = BlockBuilder(action_cap=1000, byte_cap=500, mode="highest_fee_per_action")
    txs = [_tx(1, 5000), _tx(1, 5000), _tx(1, 5000)]  # each 250 bytes
    selected = bb.select(txs, 5000)
    assert len(selected) == 2  # 500 / 250 = 2


def test_zip317_weighted_random_respects_cap():
    rng = random.Random(42)
    bb = BlockBuilder(action_cap=5, byte_cap=100000, mode="zip317_weighted_random", rng=rng)
    txs = [_tx(3, 30000), _tx(3, 15000)]
    selected = bb.select(txs, 5000)
    assert sum(tx.logical_actions for tx in selected) <= 5


def test_fifo_selects_oldest():
    bb = BlockBuilder(action_cap=5, byte_cap=100000, mode="fifo")
    tx0 = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                  logical_actions=3, byte_size=750, fee_paid=9000)
    tx1 = make_tx(kind=TxKind.HONEST, created_height=1, expiry_height=100,
                  logical_actions=3, byte_size=750, fee_paid=30000)
    selected = bb.select([tx1, tx0], 5000)
    assert len(selected) == 1
    assert selected[0].created_height == 0  # oldest first
