"""Tests for oracle fee computation."""

from zfeesim.oracle import (
    weighted_median,
    transaction_weighted_median_fee,
    action_weighted_median_fee,
    capped_effective_fee_median,
    quantize_power_of_10,
)
from zfeesim.tx import make_tx
from zfeesim.types import Block, TxKind


def _block(height: int, txs_data: list[tuple[int, int, int]]) -> Block:
    """txs_data: list of (actions, fee_paid, kind_is_attacker)"""
    txs = []
    for actions, fee, is_atk in txs_data:
        kind = TxKind.ATTACKER if is_atk else TxKind.HONEST
        txs.append(make_tx(kind=kind, created_height=height, expiry_height=height + 40,
                           logical_actions=actions, byte_size=actions * 250, fee_paid=fee))
    return Block(height=height, txs=txs, marginal_fee=5000)


def test_weighted_median_basic():
    assert weighted_median([(10.0, 1), (20.0, 1), (30.0, 1)]) == 20.0


def test_weighted_median_action_weighted():
    # 5000 fpa with 10 actions, 50000 fpa with 1 action -> median weighted by actions
    result = weighted_median([(5000.0, 10), (50000.0, 1)])
    assert result == 5000.0  # 10 actions at 5000 dominate


def test_transaction_weighted_median():
    blocks = [_block(0, [(3, 15000, 0), (3, 15000, 0), (1, 50000, 0)])]
    result = transaction_weighted_median_fee(blocks)
    assert result == 15000 / 3  # median of [5000, 5000, 50000] = 5000


def test_action_weighted_vs_transaction_weighted():
    # 2 txs with 1 action at high fee, 1 tx with 10 actions at low fee
    blocks = [_block(0, [(1, 50000, 0), (1, 50000, 0), (10, 50000, 0)])]
    tw = transaction_weighted_median_fee(blocks)
    aw = action_weighted_median_fee(blocks)
    # Transaction-weighted: median of 3 txs -> middle one
    # Action-weighted: 10 actions dominate -> lower fee/action
    assert aw == 50000 / 10  # 10-action tx has lower fee/action, dominates


def test_quantize_power_of_10():
    assert quantize_power_of_10(5000) == 1000   # floor(log10(5000))=3 -> 1000
    assert quantize_power_of_10(3000) == 1000
    assert quantize_power_of_10(10000) == 10000
    assert quantize_power_of_10(0) == 0


def test_capped_effective_fee():
    # High fee tx: cap should limit its contribution
    blocks = [_block(0, [(2, 200000, 0)])]  # 100000 fpa, but cf=10000, cap=4x=40000
    capped = capped_effective_fee_median(blocks, 5000)
    uncapped = action_weighted_median_fee(blocks)
    assert capped < uncapped
