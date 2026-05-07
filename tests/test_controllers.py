"""Tests for fee controllers."""

from zfeesim.controllers import (
    FixedZip317Controller,
    ComparableMedianController,
    ComparableMedianHysteresisController,
    BinaryFastLaneController,
    AIMDBucketController,
    _quantize_aimd_multiplier,
)
from zfeesim.tx import make_tx
from zfeesim.types import Block, TxKind


def _block_with_fee(height: int, fee_per_action: int, actions: int = 3) -> Block:
    tx = make_tx(kind=TxKind.HONEST, created_height=height, expiry_height=height + 40,
                 logical_actions=actions, byte_size=actions * 250,
                 fee_paid=fee_per_action * actions)
    return Block(height=height, txs=[tx], marginal_fee=5000)


def test_fixed_controller_constant():
    ctrl = FixedZip317Controller(marginal_fee=5000)
    assert ctrl.current_fee() == 5000
    assert ctrl.fast_lane_open() is False


def test_comparable_median_responds_to_high_fees():
    ctrl = ComparableMedianController(lookback=10, reorg_buffer=2,
                                      oracle="action_weighted_median",
                                      base_fee=5000, floor_fee=5000)
    chain = []
    # Fill with high-fee blocks
    for h in range(20):
        chain.append(_block_with_fee(h, 50000))
        ctrl.process_block(chain)
    # Fee should have risen from base
    assert ctrl.current_fee() >= 5000


def test_hysteresis_resists_immediate_jump():
    ctrl = ComparableMedianHysteresisController(
        lookback=5, reorg_buffer=1, base_fee=5000, floor_fee=5000,
        move_up_consecutive=3, move_down_consecutive=10,
    )
    chain = []
    # Just 1 high-fee block shouldn't cause jump
    chain.append(_block_with_fee(0, 50000))
    ctrl.process_block(chain)
    assert ctrl.current_fee() == 5000  # Should not have jumped yet


def test_quantize_aimd_multiplier():
    assert _quantize_aimd_multiplier(1.0) == 1
    assert _quantize_aimd_multiplier(1.5) == 2
    assert _quantize_aimd_multiplier(3.0) == 5  # 3.0 > 2.5 threshold
    assert _quantize_aimd_multiplier(5.0) == 5
    assert _quantize_aimd_multiplier(8.0) == 10


def test_binary_fast_lane_opens():
    ctrl = BinaryFastLaneController(
        base_fee=5000, open_threshold=0.5,
        synthetic_actions_per_block=100, block_action_cap=200,
    )
    # Block with no synthetic -> high displacement
    tx = make_tx(kind=TxKind.HONEST, created_height=0, expiry_height=40,
                 logical_actions=100, byte_size=25000, fee_paid=500000)
    block = Block(height=0, txs=[tx], marginal_fee=5000)
    ctrl.process_block([block])
    assert ctrl.fast_lane_open() is True
