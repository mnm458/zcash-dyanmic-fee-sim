# zfeesim — Zcash Dynamic Fee Simulator

Block-level economic simulator for evaluating Zcash dynamic fee-market designs under honest and adversarial demand. The headline metric is `harm_ratio = honest_overpayment / attacker_cost`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest  # 68 tests, ~20s
```

## CLI Commands

```bash
# Run a single scenario
python -m zfeesim.runner --scenario experiments/burst_spam_persistence.yaml

# Parameter sweep
python -m zfeesim.runner sweep \
  --base experiments/low_volume_median_poisoning.yaml \
  --param attacker.actions_per_block \
  --values 50,100,200,500

# Head-to-head comparison
python -m zfeesim.runner compare \
  --configs no_hyst=experiments/burst_spam_persistence.yaml \
            hyst=experiments/hysteresis_compare.yaml

# Validation report (sweeps + comparisons + congestion checks)
python -m zfeesim.runner validate

# Adversarial optimization (find cheapest attack per strategy)
python -m zfeesim.runner adversarial

# Full audit (cost traces + formula checks + robustness + defense probes)
python -m zfeesim.runner audit
```

All outputs go to `results/` (CSV per-block, JSON summary, markdown reports).

## Experiments

| File | Attack | Controller | Tests |
|------|--------|-----------|-------|
| `low_volume_median_poisoning.yaml` | MedianPoisoning, 300 act/blk, 10x | ComparableMedian, action-weighted | Oracle manipulation under low demand |
| `burst_spam_persistence.yaml` | BurstSpam, 500 act/blk, 10x, 10 blocks | ComparableMedian | Fee persistence after short burst |
| `fast_lane_flap.yaml` | FastLaneFlap, 90 act/blk | BinaryFastLane, threshold=0.95 | Fast-lane instability near threshold |
| `hysteresis_compare.yaml` | BucketNudging, 50 act/blk, 1.5x | ComparableMedianHysteresis | Bucket stability under nudging |
| `aimd_compare.yaml` | BurstSpam, 300 act/blk, 5x | AIMD | Smooth fee response to burst |
| `miner_self_dealing.yaml` | MinerSelfDealing, 200 act/blk, 5x | ComparableMedian | Oracle contamination via wash fees |

## Components

### Fee Controllers

| Controller | Behavior |
|-----------|----------|
| `FixedZip317Controller` | Constant marginal fee (baseline) |
| `ComparableMedianController` | Lookback oracle + power-of-10 quantization |
| `ComparableMedianWithCapController` | Same, but oracle caps per-tx contribution at 4x conventional |
| `ComparableMedianHysteresisController` | Requires N consecutive blocks above/below threshold before bucket change |
| `BinaryFastLaneController` | 1x/10x fee based on synthetic displacement ratio |
| `PriorityBucketController` | Graduated 1x/2x/5x/10x instead of binary |
| `AIMDBucketController` | Additive increase / multiplicative decrease on internal multiplier |
| `AIMDWithHysteresisController` | AIMD + hysteresis on the quantized output |

### Oracle Variants

| Oracle | Weight | Sybil resistance |
|--------|--------|-----------------|
| `transaction_weighted_median` | 1 per tx | Low — splitting is free |
| `action_weighted_median` | logical_actions per tx | Medium — splitting costs proportional actions |
| `capped_effective_fee_median` | logical_actions, fee capped at 4x conventional | High — extreme overpayment doesn't move oracle |
| `byte_share_weighted_median` | byte_size per tx | Medium — weights by capacity consumed |

### Block Builders

| Mode | Selection rule |
|------|---------------|
| `highest_fee_per_action` | Greedy by fee/action descending |
| `zip317_weighted_random` | Probabilistic, weight = min(fee/conventional, 4.0) |
| `fifo` | Oldest first |
| `random` | Uniform random |

### Attacker Strategies

| Attacker | Behavior |
|---------|----------|
| `BurstSpamAttacker` | Floods N blocks with high-fee actions, then stops |
| `MedianPoisoningAttacker` | Sustained high-fee actions to shift lookback median |
| `BucketBoundaryNudgingAttacker` | Small nudges near quantization boundaries |
| `FastLaneFlapAttacker` | Displaces synthetic demand to trigger fast-lane oscillation |
| `MinerSelfDealingAttacker` | Miner includes own high-fee txs (recoverable cost) |
| `SybilSplitAttacker` | Splits demand into many 1-action txs to exploit tx-weighted oracle |

### Wallet Policies

| Wallet | Fee behavior |
|--------|-------------|
| `patient` | Always 1x marginal |
| `normal` | 1x marginal |
| `urgent` | 10x when fast lane open, 2x otherwise |
| `exchange` | 3x always |
| `stale` | Fixed 5000 zats regardless of dynamic fee |

## YAML Config Structure

```yaml
name: scenario_name
num_blocks: 500
random_seed: 42

chain:
  block_action_cap: 1000
  block_byte_cap: 2000000

zip317:
  marginal_fee: 5000

controller:
  type: ComparableMedianController
  lookback: 50
  reorg_buffer: 5
  oracle: action_weighted_median
  quantization: power_of_10
  base_fee: 5000
  floor_fee: 5000

block_builder:
  mode: highest_fee_per_action

synthetic:
  enabled: true
  actions_per_block: 100
  fee_per_action: 5000

honest_demand:
  arrival_rate: 30
  mean_actions: 3
  expiry_blocks: 40
  urgency_distribution:
    patient: 0.5
    normal: 0.4
    urgent: 0.1

attacker:
  enabled: true
  type: MedianPoisoningAttacker
  start_height: 100
  end_height: 200
  actions_per_block: 300
  target_fee_multiplier: 10
```

## Metrics

### Headline

| Metric | Formula |
|--------|---------|
| `harm_ratio` | `honest_overpayment / effective_attacker_cost` |
| `effective_attacker_cost` | `attacker_nominal + miner_nominal - miner_recovered` |
| `honest_overpayment` | `honest_total_fee - honest_zip317_baseline_fee` |

### Cost Breakdown (zats and ZEC)

`attacker_nominal_fee_paid`, `miner_self_nominal_fee_paid`, `miner_recovered_fee`, `effective_attacker_cost`, `effective_attacker_cost_zec`, `honest_total_fee`, `honest_total_fee_zec`, `honest_overpayment_zec`

### Fee Dynamics

`raw_oracle_fee_final`, `public_fee_bucket_final`, `fee_bucket_jumps`, `fee_volatility`

### Confirmation

`median_confirmation_delay`, `p95_confirmation_delay`, `expired_honest_transactions`, `expired_attacker_transactions`

### Fast Lane

`fast_lane_open_blocks`, `fast_lane_flaps`

### Synthetic

`synthetic_displacement_ratio_avg`

### Per-wallet

`wallet_policy_fees`, `wallet_policy_delays` (median/mean/count per wallet type)

## Tests

| File | Count | What it covers |
|------|-------|---------------|
| `test_zip317.py` | 7 | Conventional fee, weight ratio, policy acceptance |
| `test_oracle.py` | 6 | Weighted median, oracle variants, quantization, capping |
| `test_block_builder.py` | 5 | Selection modes, action/byte cap enforcement |
| `test_controllers.py` | 5 | Fixed, median, hysteresis, AIMD, fast lane controllers |
| `test_adversarial_scenarios.py` | 16 | All spec hypothesis tests (1–16) |
| `test_sanity.py` | 14 | Unit scales, ZEC conversion, cost split, congestion delays |
| `test_adversarial_opt.py` | 7 | Grid search, optimizer runs, defense probe validation |
| `test_audit.py` | 8 | harm_ratio formula, cost isolation, AIMD usability |

### What each hypothesis test validates

| # | Test | Validates |
|---|------|----------|
| 1 | No congestion fixed ZIP-317 | Baseline: low demand, instant confirmation, no harm |
| 2 | Demand above capacity | Mempool growth, confirmation delays, expiry |
| 3 | Higher fee confirms faster | Fee sorting prioritizes urgent wallets |
| 4 | Weight ratio cap | 100x overpayment gives 4x priority, not 100x |
| 5 | Tx-weighted sybil sensitivity | Splitting txs cheaply moves tx-weighted median |
| 6 | Action-weighted sybil resistance | Action-weighted harder to manipulate |
| 7 | Capped oracle reduces poisoning | Cap limits extreme overpayment influence |
| 8 | Burst spam persistence | Lookback lag keeps fees elevated post-attack |
| 9 | Boundary nudging causes jumps | Small cost, large bucket jump without hysteresis |
| 10 | Hysteresis reduces flapping | Fewer jumps than without hysteresis |
| 11 | Binary fast lane flaps | Threshold instability near 0.95 |
| 12 | Fast-lane hysteresis helps | Fewer flaps with separate open/close thresholds |
| 13 | Priority buckets reduce cliff | Graduated 1/2/5/10x cheaper than binary 1/10x |
| 14 | AIMD smooths fees | Lower volatility than direct median |
| 15 | Miner self-dealing poisons oracle | Wash fees inflate future honest fees |
| 16 | Synthetic excluded from harm | Synthetic txs don't count as honest overpayment |

## Interpreting Results

`harm_ratio < 1` — attacker pays more than the harm caused. Mechanism is resilient.

`harm_ratio > 1` — attacker externalizes cost onto honest users. Mechanism is fragile under this attack.

`harm_ratio >> 1` — cheap attack, expensive harm. Serious design concern.

To check whether a defense mitigates an attack, run `adversarial` and look at the defense survival table. An attack that survives 0/4 defenses is well-handled. An attack that survives 4/4 is a fundamental vulnerability.

## Unit Reference

- 1 ZEC = 100,000,000 zatoshis (zats)
- ZIP-317 marginal fee = 5,000 zats/action = 0.00005 ZEC/action
- 2-action tx conventional fee = 10,000 zats = 0.0001 ZEC
