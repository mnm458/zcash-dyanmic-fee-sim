"""Fee controllers: determine the current marginal fee and fast-lane state."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .oracle import (
    compute_oracle_fee,
    get_lookback_blocks,
    quantize_power_of_10,
    quantize_round10,
    quantize_priority_multipliers,
    PRIORITY_MULTIPLIERS,
)
from .types import Block


class FeeController(ABC):
    @abstractmethod
    def current_fee(self) -> int: ...

    @abstractmethod
    def fast_lane_open(self) -> bool: ...

    @abstractmethod
    def process_block(self, chain: list[Block]) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...


class FixedZip317Controller(FeeController):
    def __init__(self, marginal_fee: int = 5000) -> None:
        self._fee = marginal_fee

    def current_fee(self) -> int:
        return self._fee

    def fast_lane_open(self) -> bool:
        return False

    def process_block(self, chain: list[Block]) -> None:
        pass

    def reset(self) -> None:
        pass


class ComparableMedianController(FeeController):
    def __init__(self, *, lookback: int = 50, reorg_buffer: int = 5,
                 oracle: str = "action_weighted_median",
                 quantization: str = "power_of_10",
                 base_fee: int = 5000,
                 floor_fee: int = 5000,
                 oracle_include_synthetic: bool = True) -> None:
        self.lookback = lookback
        self.reorg_buffer = reorg_buffer
        self.oracle_type = oracle
        self.quantization = quantization
        self.base_fee = base_fee
        self.floor_fee = floor_fee
        self.include_synthetic = oracle_include_synthetic
        self._fee = base_fee
        self._raw_fee: float = float(base_fee)
        self._fast_lane = False

    def current_fee(self) -> int:
        return self._fee

    def fast_lane_open(self) -> bool:
        return self._fast_lane

    def process_block(self, chain: list[Block]) -> None:
        lb = get_lookback_blocks(chain, self.lookback, self.reorg_buffer)
        if not lb:
            return
        raw = compute_oracle_fee(lb, self.oracle_type, self._fee,
                                 include_synthetic=self.include_synthetic)
        self._raw_fee = raw
        if self.quantization == "round10":
            self._fee = max(self.floor_fee, quantize_round10(raw))
        elif self.quantization == "power_of_10":
            self._fee = max(self.floor_fee, quantize_power_of_10(raw))
        elif self.quantization == "fixed_priority_multipliers":
            self._fee = max(self.floor_fee, quantize_priority_multipliers(raw, self.base_fee))
        else:
            self._fee = max(self.floor_fee, int(raw))

    def reset(self) -> None:
        self._fee = self.base_fee
        self._raw_fee = float(self.base_fee)
        self._fast_lane = False

    @property
    def raw_oracle_fee(self) -> float:
        return self._raw_fee


class ComparableMedianWithCapController(ComparableMedianController):
    def process_block(self, chain: list[Block]) -> None:
        lb = get_lookback_blocks(chain, self.lookback, self.reorg_buffer)
        if not lb:
            return
        raw = compute_oracle_fee(lb, "capped_effective_fee_median", self._fee,
                                 include_synthetic=self.include_synthetic)
        self._raw_fee = raw
        if self.quantization == "round10":
            self._fee = max(self.floor_fee, quantize_round10(raw))
        else:
            self._fee = max(self.floor_fee, quantize_power_of_10(raw))


class ComparableMedianHysteresisController(FeeController):
    def __init__(self, *, lookback: int = 50, reorg_buffer: int = 5,
                 oracle: str = "action_weighted_median",
                 base_fee: int = 5000, floor_fee: int = 5000,
                 move_up_ratio: float = 0.8,
                 move_down_ratio: float = 0.2,
                 move_up_consecutive: int = 5,
                 move_down_consecutive: int = 20,
                 oracle_include_synthetic: bool = True) -> None:
        self.lookback = lookback
        self.reorg_buffer = reorg_buffer
        self.oracle_type = oracle
        self.base_fee = base_fee
        self.floor_fee = floor_fee
        self.include_synthetic = oracle_include_synthetic
        self.move_up_ratio = move_up_ratio
        self.move_down_ratio = move_down_ratio
        self.move_up_consecutive = move_up_consecutive
        self.move_down_consecutive = move_down_consecutive

        self._bucket = base_fee
        self._raw_fee: float = float(base_fee)
        self._up_count = 0
        self._down_count = 0

    def current_fee(self) -> int:
        return self._bucket

    def fast_lane_open(self) -> bool:
        return False

    def process_block(self, chain: list[Block]) -> None:
        lb = get_lookback_blocks(chain, self.lookback, self.reorg_buffer)
        if not lb:
            return
        raw = compute_oracle_fee(lb, self.oracle_type, self._bucket,
                                 include_synthetic=self.include_synthetic)
        self._raw_fee = raw

        next_up = self._bucket * 10
        next_down = max(self.floor_fee, self._bucket // 10)
        up_threshold = next_up * self.move_up_ratio
        down_threshold = next_down + (self._bucket - next_down) * self.move_down_ratio

        if raw >= up_threshold:
            self._up_count += 1
            self._down_count = 0
        elif raw <= down_threshold:
            self._down_count += 1
            self._up_count = 0
        else:
            self._up_count = 0
            self._down_count = 0

        if self._up_count >= self.move_up_consecutive:
            self._bucket = next_up
            self._up_count = 0
            self._down_count = 0
        elif self._down_count >= self.move_down_consecutive:
            self._bucket = next_down
            self._up_count = 0
            self._down_count = 0

    def reset(self) -> None:
        self._bucket = self.base_fee
        self._raw_fee = float(self.base_fee)
        self._up_count = 0
        self._down_count = 0

    @property
    def raw_oracle_fee(self) -> float:
        return self._raw_fee


class BinaryFastLaneController(FeeController):
    def __init__(self, *, base_fee: int = 5000,
                 fast_lane_multiplier: int = 10,
                 open_threshold: float = 0.95,
                 close_threshold: float = 0.95,
                 use_hysteresis: bool = False,
                 open_consecutive: int = 1,
                 close_consecutive: int = 1,
                 synthetic_actions_per_block: int = 100,
                 block_action_cap: int = 1000) -> None:
        self._base_fee = base_fee
        self._fast_lane_multiplier = fast_lane_multiplier
        self._open_threshold = open_threshold
        self._close_threshold = close_threshold
        self._use_hysteresis = use_hysteresis
        self._open_consecutive = open_consecutive
        self._close_consecutive = close_consecutive
        self._synthetic_per_block = synthetic_actions_per_block
        self._block_action_cap = block_action_cap
        self._fast_lane = False
        self._open_count = 0
        self._close_count = 0

    def current_fee(self) -> int:
        if self._fast_lane:
            return self._base_fee * self._fast_lane_multiplier
        return self._base_fee

    def fast_lane_open(self) -> bool:
        return self._fast_lane

    def process_block(self, chain: list[Block]) -> None:
        if not chain:
            return
        block = chain[-1]
        from .types import TxKind
        synth_included = sum(
            tx.logical_actions for tx in block.txs if tx.kind == TxKind.SYNTHETIC
        )
        ratio = 1.0 - (synth_included / self._synthetic_per_block) if self._synthetic_per_block > 0 else 0.0
        ratio = max(0.0, min(1.0, ratio))

        if self._use_hysteresis:
            if not self._fast_lane:
                if ratio >= self._open_threshold:
                    self._open_count += 1
                else:
                    self._open_count = 0
                if self._open_count >= self._open_consecutive:
                    self._fast_lane = True
                    self._open_count = 0
                    self._close_count = 0
            else:
                if ratio < self._close_threshold:
                    self._close_count += 1
                else:
                    self._close_count = 0
                if self._close_count >= self._close_consecutive:
                    self._fast_lane = False
                    self._close_count = 0
                    self._open_count = 0
        else:
            self._fast_lane = ratio >= self._open_threshold

    def reset(self) -> None:
        self._fast_lane = False
        self._open_count = 0
        self._close_count = 0


class PriorityBucketController(FeeController):
    def __init__(self, *, base_fee: int = 5000,
                 buckets: list[int] | None = None,
                 open_threshold: float = 0.95,
                 synthetic_actions_per_block: int = 100) -> None:
        self._base_fee = base_fee
        self._buckets = buckets or [1, 2, 5, 10]
        self._open_threshold = open_threshold
        self._synthetic_per_block = synthetic_actions_per_block
        self._active_multiplier = 1
        self._displacement_ratio = 0.0

    def current_fee(self) -> int:
        return self._base_fee * self._active_multiplier

    def fast_lane_open(self) -> bool:
        return self._active_multiplier > 1

    def process_block(self, chain: list[Block]) -> None:
        if not chain:
            return
        block = chain[-1]
        from .types import TxKind
        synth_included = sum(
            tx.logical_actions for tx in block.txs if tx.kind == TxKind.SYNTHETIC
        )
        self._displacement_ratio = 1.0 - (synth_included / self._synthetic_per_block) if self._synthetic_per_block > 0 else 0.0
        self._displacement_ratio = max(0.0, min(1.0, self._displacement_ratio))

        if self._displacement_ratio >= self._open_threshold:
            for m in self._buckets:
                if m > self._active_multiplier:
                    self._active_multiplier = m
                    break
        else:
            if self._active_multiplier > 1:
                idx = self._buckets.index(self._active_multiplier) if self._active_multiplier in self._buckets else 0
                if idx > 0:
                    self._active_multiplier = self._buckets[idx - 1]
                else:
                    self._active_multiplier = 1

    def reset(self) -> None:
        self._active_multiplier = 1
        self._displacement_ratio = 0.0


def _quantize_aimd_multiplier(m: float) -> int:
    if m <= 1.25:
        return 1
    if m <= 2.5:
        return 2
    if m <= 7.5:
        return 5
    return 10


class AIMDBucketController(FeeController):
    def __init__(self, *, base_fee: int = 5000,
                 alpha: float = 0.25, beta: float = 0.90,
                 target_utilization: float = 0.70,
                 lower_utilization: float = 0.40,
                 increase_window: int = 5,
                 decrease_window: int = 20,
                 block_action_cap: int = 1000) -> None:
        self._base_fee = base_fee
        self._alpha = alpha
        self._beta = beta
        self._target_util = target_utilization
        self._lower_util = lower_utilization
        self._increase_window = increase_window
        self._decrease_window = decrease_window
        self._block_action_cap = block_action_cap
        self._multiplier = 1.0
        self._above_count = 0
        self._below_count = 0

    def current_fee(self) -> int:
        q = _quantize_aimd_multiplier(self._multiplier)
        return self._base_fee * q

    def fast_lane_open(self) -> bool:
        return self._multiplier > 1.25

    def process_block(self, chain: list[Block]) -> None:
        if not chain:
            return
        block = chain[-1]
        from .types import TxKind
        real_actions = sum(tx.logical_actions for tx in block.txs if tx.kind != TxKind.SYNTHETIC)
        util = real_actions / self._block_action_cap if self._block_action_cap > 0 else 0.0

        if util > self._target_util:
            self._above_count += 1
            self._below_count = 0
        elif util < self._lower_util:
            self._below_count += 1
            self._above_count = 0
        else:
            self._above_count = 0
            self._below_count = 0

        if self._above_count >= self._increase_window:
            self._multiplier += self._alpha
            self._above_count = 0
        if self._below_count >= self._decrease_window:
            self._multiplier *= self._beta
            self._below_count = 0

        self._multiplier = max(1.0, min(self._multiplier, 10.0))

    def reset(self) -> None:
        self._multiplier = 1.0
        self._above_count = 0
        self._below_count = 0

    @property
    def raw_multiplier(self) -> float:
        return self._multiplier


class AIMDWithHysteresisController(AIMDBucketController):
    def __init__(self, *, move_up_consecutive: int = 5,
                 move_down_consecutive: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hyst_up = move_up_consecutive
        self._hyst_down = move_down_consecutive
        self._prev_q = 1
        self._q_up_count = 0
        self._q_down_count = 0

    def current_fee(self) -> int:
        return self._base_fee * self._prev_q

    def process_block(self, chain: list[Block]) -> None:
        super().process_block(chain)
        candidate_q = _quantize_aimd_multiplier(self._multiplier)

        if candidate_q > self._prev_q:
            self._q_up_count += 1
            self._q_down_count = 0
            if self._q_up_count >= self._hyst_up:
                self._prev_q = candidate_q
                self._q_up_count = 0
        elif candidate_q < self._prev_q:
            self._q_down_count += 1
            self._q_up_count = 0
            if self._q_down_count >= self._hyst_down:
                self._prev_q = candidate_q
                self._q_down_count = 0
        else:
            self._q_up_count = 0
            self._q_down_count = 0

    def reset(self) -> None:
        super().reset()
        self._prev_q = 1
        self._q_up_count = 0
        self._q_down_count = 0


CONTROLLER_REGISTRY: dict[str, type] = {
    "FixedZip317Controller": FixedZip317Controller,
    "ComparableMedianController": ComparableMedianController,
    "ComparableMedianWithCapController": ComparableMedianWithCapController,
    "ComparableMedianHysteresisController": ComparableMedianHysteresisController,
    "BinaryFastLaneController": BinaryFastLaneController,
    "PriorityBucketController": PriorityBucketController,
    "AIMDBucketController": AIMDBucketController,
    "AIMDWithHysteresisController": AIMDWithHysteresisController,
}
