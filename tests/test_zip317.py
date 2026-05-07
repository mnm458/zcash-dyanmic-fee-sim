"""Tests for ZIP-317 fee logic."""

from zfeesim.tx import make_tx
from zfeesim.types import TxKind
from zfeesim.zip317 import conventional_fee, weight_ratio, policy_accepts, GRACE_ACTIONS


def _tx(actions: int, fee: int) -> "Tx":
    from zfeesim.types import Tx
    return make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=100,
                   logical_actions=actions, byte_size=actions * 250, fee_paid=fee)


def test_conventional_fee_grace():
    tx = _tx(1, 5000)
    assert conventional_fee(tx, 5000) == 5000 * GRACE_ACTIONS


def test_conventional_fee_above_grace():
    tx = _tx(5, 25000)
    assert conventional_fee(tx, 5000) == 5000 * 5


def test_weight_ratio_exact_conventional():
    tx = _tx(2, 10000)
    assert weight_ratio(tx, 5000) == 1.0


def test_weight_ratio_capped():
    tx = _tx(2, 100000)
    assert weight_ratio(tx, 5000) == 4.0


def test_weight_ratio_underpay():
    tx = _tx(2, 5000)
    assert weight_ratio(tx, 5000) == 0.5


def test_policy_accepts_exact():
    tx = _tx(2, 10000)
    assert policy_accepts(tx, 5000) is True


def test_policy_rejects_underpay():
    tx = _tx(2, 5000)
    assert policy_accepts(tx, 5000) is False
