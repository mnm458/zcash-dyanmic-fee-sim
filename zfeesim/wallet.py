"""Wallet fee-bidding policies."""

from __future__ import annotations

from .types import Tx


class WalletPolicy:
    name: str = "normal"

    def choose_fee(self, current_fee: int, fast_lane_open: bool,
                   mempool_snapshot: list[Tx] | None = None) -> int:
        return current_fee


class PatientWallet(WalletPolicy):
    name = "patient"

    def choose_fee(self, current_fee: int, fast_lane_open: bool,
                   mempool_snapshot: list[Tx] | None = None) -> int:
        return current_fee  # always 1x


class NormalWallet(WalletPolicy):
    name = "normal"

    def choose_fee(self, current_fee: int, fast_lane_open: bool,
                   mempool_snapshot: list[Tx] | None = None) -> int:
        return current_fee


class UrgentWallet(WalletPolicy):
    name = "urgent"

    def __init__(self, multiplier: int = 10) -> None:
        self.multiplier = multiplier

    def choose_fee(self, current_fee: int, fast_lane_open: bool,
                   mempool_snapshot: list[Tx] | None = None) -> int:
        if fast_lane_open:
            return current_fee * self.multiplier
        return current_fee * 2  # slightly above marginal even when no fast lane


class ExchangeWallet(WalletPolicy):
    name = "exchange"

    def choose_fee(self, current_fee: int, fast_lane_open: bool,
                   mempool_snapshot: list[Tx] | None = None) -> int:
        return current_fee * 3  # conservative overpay


class StaleWallet(WalletPolicy):
    name = "stale"

    def __init__(self, fixed_fee: int = 5000) -> None:
        self.fixed_fee = fixed_fee

    def choose_fee(self, current_fee: int, fast_lane_open: bool,
                   mempool_snapshot: list[Tx] | None = None) -> int:
        return self.fixed_fee  # ignores dynamic fee


WALLET_REGISTRY: dict[str, WalletPolicy] = {
    "patient": PatientWallet(),
    "normal": NormalWallet(),
    "urgent": UrgentWallet(),
    "exchange": ExchangeWallet(),
    "stale": StaleWallet(),
}


def get_wallet(name: str) -> WalletPolicy:
    return WALLET_REGISTRY.get(name, NormalWallet())
