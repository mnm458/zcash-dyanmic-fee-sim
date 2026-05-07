"""Metrics collection for simulation runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .oracle import ZATS_PER_ZEC, zats_to_zec
from .types import Block, Tx, TxKind


@dataclass
class HeightRecord:
    height: int = 0
    marginal_fee: int = 0
    raw_oracle_fee: float = 0.0
    public_fee_bucket: int = 0
    fast_lane_open: bool = False
    honest_actions_included: int = 0
    attacker_actions_included: int = 0
    synthetic_actions_included: int = 0
    miner_self_actions_included: int = 0
    total_actions_included: int = 0
    honest_fee_paid: int = 0
    attacker_fee_paid: int = 0
    miner_self_fee_paid: int = 0
    mempool_size: int = 0
    mempool_honest_actions: int = 0
    mempool_attacker_actions: int = 0
    expired_honest: int = 0
    expired_attacker: int = 0
    synthetic_displacement_ratio: float = 0.0
    # Oracle sample breakdown
    honest_actions_in_oracle: int = 0
    attacker_actions_in_oracle: int = 0
    synthetic_actions_in_oracle: int = 0
    miner_self_actions_in_oracle: int = 0
    total_actions_in_oracle: int = 0


class MetricsCollector:
    def __init__(self, zip317_marginal_fee: int = 5000,
                 synthetic_actions_per_block: int = 100,
                 miner_fee_recovery_rate: float = 0.0,
                 baseline_overpayment: int = 0) -> None:
        self.zip317_fee = zip317_marginal_fee
        self.synthetic_per_block = synthetic_actions_per_block
        self.miner_recovery_rate = miner_fee_recovery_rate
        self._baseline_overpayment = baseline_overpayment
        self.records: list[HeightRecord] = []
        self._honest_confirmation_delays: list[int] = []
        self._attacker_nominal_fee: int = 0
        self._miner_self_nominal_fee: int = 0
        self._honest_total_fee: int = 0
        self._honest_zip317_fee: int = 0
        self._total_expired_honest: int = 0
        self._total_expired_attacker: int = 0
        self._fee_buckets: list[int] = []
        self._fast_lane_states: list[bool] = []
        self._wallet_fees: dict[str, list[int]] = {}
        self._wallet_delays: dict[str, list[int]] = {}

    def record_block(self, block: Block, mempool_size: int,
                     mempool_honest_actions: int, mempool_attacker_actions: int,
                     expired_honest: list[Tx], expired_attacker: list[Tx],
                     oracle_breakdown: dict | None = None) -> None:
        r = HeightRecord(height=block.height)
        r.marginal_fee = block.marginal_fee
        r.raw_oracle_fee = block.raw_oracle_fee or 0.0
        r.public_fee_bucket = block.public_fee_bucket or block.marginal_fee
        r.fast_lane_open = block.fast_lane_open

        for tx in block.txs:
            delay = block.height - tx.created_height
            if tx.kind == TxKind.HONEST:
                r.honest_actions_included += tx.logical_actions
                r.honest_fee_paid += tx.fee_paid
                self._honest_total_fee += tx.fee_paid
                self._honest_zip317_fee += self.zip317_fee * max(2, tx.logical_actions)
                self._honest_confirmation_delays.append(delay)
                self._wallet_fees.setdefault(tx.wallet_policy, []).append(tx.fee_paid)
                self._wallet_delays.setdefault(tx.wallet_policy, []).append(delay)
            elif tx.kind == TxKind.ATTACKER:
                r.attacker_actions_included += tx.logical_actions
                r.attacker_fee_paid += tx.fee_paid
                self._attacker_nominal_fee += tx.fee_paid
            elif tx.kind == TxKind.MINER_SELF:
                r.miner_self_actions_included += tx.logical_actions
                r.miner_self_fee_paid += tx.fee_paid
                self._miner_self_nominal_fee += tx.fee_paid
            elif tx.kind == TxKind.SYNTHETIC:
                r.synthetic_actions_included += tx.logical_actions

        r.total_actions_included = sum(tx.logical_actions for tx in block.txs)
        r.mempool_size = mempool_size
        r.mempool_honest_actions = mempool_honest_actions
        r.mempool_attacker_actions = mempool_attacker_actions
        r.expired_honest = len(expired_honest)
        r.expired_attacker = len(expired_attacker)

        self._total_expired_honest += len(expired_honest)
        self._total_expired_attacker += len(expired_attacker)

        if self.synthetic_per_block > 0:
            r.synthetic_displacement_ratio = 1.0 - (r.synthetic_actions_included / self.synthetic_per_block)
        r.synthetic_displacement_ratio = max(0.0, min(1.0, r.synthetic_displacement_ratio))

        if oracle_breakdown:
            r.honest_actions_in_oracle = oracle_breakdown.get("honest_actions_in_oracle", 0)
            r.attacker_actions_in_oracle = oracle_breakdown.get("attacker_actions_in_oracle", 0)
            r.synthetic_actions_in_oracle = oracle_breakdown.get("synthetic_actions_in_oracle", 0)
            r.miner_self_actions_in_oracle = oracle_breakdown.get("miner_self_actions_in_oracle", 0)
            r.total_actions_in_oracle = oracle_breakdown.get("total_actions_in_oracle", 0)

        self._fee_buckets.append(r.public_fee_bucket)
        self._fast_lane_states.append(r.fast_lane_open)
        self.records.append(r)

    # ------------------------------------------------------------------
    # Derived cost breakdown
    # ------------------------------------------------------------------

    @property
    def attacker_nominal_fee(self) -> int:
        return self._attacker_nominal_fee

    @property
    def miner_recovered_fee(self) -> int:
        return int(self._miner_self_nominal_fee * self.miner_recovery_rate)

    @property
    def effective_attacker_cost(self) -> int:
        """nominal attacker fee + miner nominal fee - miner recovered fee."""
        return self._attacker_nominal_fee + self._miner_self_nominal_fee - self.miner_recovered_fee

    @property
    def honest_overpayment(self) -> int:
        return max(0, self._honest_total_fee - self._honest_zip317_fee)

    @property
    def incremental_overpayment(self) -> int:
        """Overpayment attributable to the attacker, not baseline mechanism overhead."""
        return max(0, self.honest_overpayment - self._baseline_overpayment)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        overpayment = self.honest_overpayment
        incremental = self.incremental_overpayment
        eff_cost = self.effective_attacker_cost
        harm_ratio = incremental / eff_cost if eff_cost > 0 else 0.0

        delays = self._honest_confirmation_delays
        if delays:
            median_delay = float(statistics.median(delays))
            sorted_d = sorted(delays)
            p95_idx = int(len(sorted_d) * 0.95)
            p95_delay = float(sorted_d[min(p95_idx, len(sorted_d) - 1)])
        else:
            median_delay = 0.0
            p95_delay = 0.0

        # Fee bucket jumps
        jumps = 0
        for i in range(1, len(self._fee_buckets)):
            if self._fee_buckets[i] != self._fee_buckets[i - 1]:
                jumps += 1

        # Fee volatility
        fee_volatility = 0.0
        if len(self._fee_buckets) > 1:
            log_changes = []
            for i in range(1, len(self._fee_buckets)):
                if self._fee_buckets[i] > 0 and self._fee_buckets[i - 1] > 0:
                    log_changes.append(
                        abs(math.log10(self._fee_buckets[i]) - math.log10(self._fee_buckets[i - 1]))
                    )
            if log_changes:
                fee_volatility = statistics.mean(log_changes)

        # Fast lane
        fl_open = sum(1 for s in self._fast_lane_states if s)
        fl_flaps = 0
        for i in range(1, len(self._fast_lane_states)):
            if self._fast_lane_states[i] != self._fast_lane_states[i - 1]:
                fl_flaps += 1

        disp_ratios = [r.synthetic_displacement_ratio for r in self.records]
        disp_avg = statistics.mean(disp_ratios) if disp_ratios else 0.0

        miner_revenue = sum(r.honest_fee_paid + r.attacker_fee_paid + r.miner_self_fee_paid
                            for r in self.records)

        raw_final = self.records[-1].raw_oracle_fee if self.records else 0.0
        bucket_final = self.records[-1].public_fee_bucket if self.records else 0

        # Per-wallet delay summaries
        wallet_delays_summary: dict[str, dict] = {}
        for wp, dlist in self._wallet_delays.items():
            if dlist:
                wallet_delays_summary[wp] = {
                    "median_delay": round(float(statistics.median(dlist)), 2),
                    "mean_delay": round(float(statistics.mean(dlist)), 2),
                    "count": len(dlist),
                }

        return {
            # ---- attacker cost split ----
            "attacker_nominal_fee_paid": self._attacker_nominal_fee,
            "miner_self_nominal_fee_paid": self._miner_self_nominal_fee,
            "miner_recovered_fee": self.miner_recovered_fee,
            "effective_attacker_cost": eff_cost,
            # keep legacy key for backward compat
            "attacker_cost": eff_cost,

            # ---- honest ----
            "honest_total_fee": self._honest_total_fee,
            "honest_overpayment_vs_fixed_zip317": overpayment,
            "baseline_overpayment": self._baseline_overpayment,
            "incremental_overpayment": incremental,
            "harm_ratio": round(harm_ratio, 4),

            # ---- ZEC equivalents ----
            "effective_attacker_cost_zec": round(zats_to_zec(eff_cost), 6),
            "honest_overpayment_zec": round(zats_to_zec(overpayment), 6),
            "incremental_overpayment_zec": round(zats_to_zec(incremental), 6),
            "honest_total_fee_zec": round(zats_to_zec(self._honest_total_fee), 6),

            # ---- confirmation ----
            "median_confirmation_delay": round(median_delay, 2),
            "p95_confirmation_delay": round(p95_delay, 2),
            "expired_honest_transactions": self._total_expired_honest,
            "expired_attacker_transactions": self._total_expired_attacker,

            # ---- fee dynamics ----
            "raw_oracle_fee_final": round(raw_final, 2),
            "public_fee_bucket_final": bucket_final,
            "fee_bucket_jumps": jumps,
            "fee_volatility": round(fee_volatility, 6),

            # ---- fast lane ----
            "fast_lane_open_blocks": fl_open,
            "fast_lane_flaps": fl_flaps,

            # ---- synthetic ----
            "synthetic_displacement_ratio_avg": round(disp_avg, 4),

            # ---- revenue ----
            "miner_revenue": miner_revenue,
            "miner_self_dealing_profit": self._miner_self_nominal_fee,

            # ---- wallet breakdowns ----
            "wallet_policy_fees": {k: sum(v) for k, v in self._wallet_fees.items()},
            "wallet_policy_delays": wallet_delays_summary,
        }

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            return
        fieldnames = list(HeightRecord.__dataclass_fields__.keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                row = {k: getattr(r, k) for k in fieldnames}
                writer.writerow(row)

    def write_debug_csv(self, path: str | Path) -> None:
        """Write per-block debug trace with all diagnostic columns."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            return

        fieldnames = [
            "height",
            "honest_actions_included", "attacker_actions_included",
            "synthetic_actions_included",
            "synthetic_displacement_ratio",
            "oracle_real_sample_count", "oracle_synthetic_sample_count",
            "raw_oracle_fee", "public_fee_bucket",
            "fast_lane_open", "marginal_fee",
            "honest_fee_paid", "attacker_fee_paid",
            "cumulative_honest_overpayment", "cumulative_attacker_cost",
            "harm_ratio",
            "mempool_size", "mempool_honest_actions", "mempool_attacker_actions",
            "expired_honest", "expired_attacker",
        ]

        cum_honest_overpayment = 0
        cum_attacker_cost = 0
        cum_honest_zip317 = 0

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                cum_honest_overpayment += r.honest_fee_paid
                # Approximate baseline per block from records
                # (actual per-tx baseline tracked internally)
                cum_attacker_cost += r.attacker_fee_paid + r.miner_self_fee_paid
                oracle_real = (r.honest_actions_in_oracle
                               + r.attacker_actions_in_oracle
                               + r.miner_self_actions_in_oracle)
                hr = (self.honest_overpayment / max(1, cum_attacker_cost)
                       if cum_attacker_cost > 0 else 0.0)
                writer.writerow({
                    "height": r.height,
                    "honest_actions_included": r.honest_actions_included,
                    "attacker_actions_included": r.attacker_actions_included,
                    "synthetic_actions_included": r.synthetic_actions_included,
                    "synthetic_displacement_ratio": round(r.synthetic_displacement_ratio, 4),
                    "oracle_real_sample_count": oracle_real,
                    "oracle_synthetic_sample_count": r.synthetic_actions_in_oracle,
                    "raw_oracle_fee": round(r.raw_oracle_fee, 2),
                    "public_fee_bucket": r.public_fee_bucket,
                    "fast_lane_open": r.fast_lane_open,
                    "marginal_fee": r.marginal_fee,
                    "honest_fee_paid": r.honest_fee_paid,
                    "attacker_fee_paid": r.attacker_fee_paid,
                    "cumulative_honest_overpayment": self.honest_overpayment,
                    "cumulative_attacker_cost": cum_attacker_cost,
                    "harm_ratio": round(hr, 4),
                    "mempool_size": r.mempool_size,
                    "mempool_honest_actions": r.mempool_honest_actions,
                    "mempool_attacker_actions": r.mempool_attacker_actions,
                    "expired_honest": r.expired_honest,
                    "expired_attacker": r.expired_attacker,
                })

    def write_summary_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
