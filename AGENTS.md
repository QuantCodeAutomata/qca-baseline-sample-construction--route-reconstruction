# Project: WETH-USDC DEX Routing Experiments

## Overview
Empirical analysis of same-pair routing behaviour across a 4-pool WETH-USDC universe
(Uniswap V2 30bps + Uniswap V3 5/30/100bps). Computes concentration metrics and
execution-quality benchmarks (SCO, FVO, G-FVO) from blockchain swap data.

## Directory Layout
```
src/
  config.py          – Pool universe (POOL_UNIVERSE dict), router labels, gas constants
  amm_math.py        – V2/V3 AMM output, marginal rate, and inverse-marginal functions
  concentration.py   – compute_activation_rate, compute_entropy, compute_hhi
  router_labels.py   – get_router_label(addr), label_transactions(df)
  filters.py         – filter_counterflow, filter_min_usdc, apply_all_filters
  synthetic_data.py  – Generate synthetic WETH-USDC swap data for testing

exp/
  exp1_baseline.py   – Baseline sample construction + concentration metrics
  exp2_benchmarks.py – SCO / FVO / G-FVO optimal routing benchmarks
  exp3_staleness.py  – State-staleness degradation analysis (N-block lag)
  exp4_trade_size.py – Size-bin stratified shortfall analysis
  exp5_sandwich.py   – Sandwich attack detection and shortfall comparison

results/             – CSV outputs from run_experiment() functions
data/                – raw / interim / fixtures blockchain data
tests/
```

## Key src Module APIs

### src/config.py
- `POOL_UNIVERSE: dict[int, dict]` – keys 0-3, each has address/protocol/fee_bps/name
- `ROUTER_LABELS: dict[str, str]` – lowercase address → label
- `BLOCK_START / BLOCK_END`: 19_500_000 / 23_000_000
- `GAS_COSTS_ETH: dict[int, int]` – gas units per pool (NOT USDC)
- `GAS_ETH_PRICE_WEI: int` – default 20 gwei
- Does NOT export: POOL_IDS, POOL_ADDRESSES, POOL_FEE_BPS, GAS_COSTS_USDC, ROUTER_ADDRESSES

### src/amm_math.py
- `v2_total_output(q, reserve_in, reserve_out, fee_bps=30)` → float
- `v3_total_output(q, liquidity, sqrt_price, token_in_is_token0: bool, fee_bps)` → float
  ⚠️  token_in_is_token0 is a **bool**, NOT a direction string; sqrt_price is raw sqrt_price_x96
- `v2_marginal_output(q, reserve_in, reserve_out, fee_bps)` → float
- `v3_marginal_output(q, liquidity, sqrt_price, token_in_is_token0, fee_bps)` → float
- `v2_inverse_marginal_price(lam, reserve_in, reserve_out, fee_bps)` → float
- `v3_inverse_marginal_price(lam, liquidity, sqrt_price, token_in_is_token0, fee_bps)` → float
- `v3_simulate_swap(amount_in, pool_state, token_in_is_token0)` → (amount_out, new_state)

### src/concentration.py
- `compute_activation_rate(x, num_pools=4)` → float  (note: num_pools kwarg)
- `compute_entropy(x)` → float (Shannon entropy, nats)
- `compute_hhi(x)` → float (sum of squared shares)
- `compute_flow_shares(q)` → np.ndarray

### src/router_labels.py
- `get_router_label(to_address)` → Optional[str]  (NOT `label_router`)
- `label_transactions(df, address_col='to_address')` → df with router_label column
- Internally uses `from config import ROUTER_LABELS` (relative import, may fail if run
  from project root with src as package)

### src/filters.py
- `filter_counterflow(df)` → df
- `filter_min_usdc(df, min_usdc=100.0)` → df
- `apply_all_filters(df, min_usdc=100.0)` → (filtered_df, stats_dict)
  Does NOT export `apply_filters`

## Synthetic Data Generator (src/synthetic_data.py)

### Pool universe constants (fallback when src.config unavailable)
- Pool 0: V2, 30bps. Pool 1: V3, 5bps. Pool 2: V3, 30bps. Pool 3: V3, 100bps
- V2 base reserves: 5000 WETH / 15M USDC
- V3 liquidity: {1: 5e22, 2: 2e22, 3: 5e21} (token√ units)
- sqrt_price_x96 = sqrt(price_USDC_per_WETH) × 2^96

### Routing mix (targets HHI ≈ 0.971)
- 72% single-pool, 20% 2-pool, 6% 3-pool, 2% 4-pool
- Multi-pool shares from Dirichlet(α=0.1) → very skewed
- USDC sizes log-normal: μ=8.607, σ=2.0 → mean ≈ 40,437 USDC
- 55% WETH→USDC, 45% USDC→WETH

### Pool state block sampling
- 1000 states sampled over 19.5M–23M range → step ≈ 3500 blocks
- Staleness analysis (exp3) only shows degradation when N > step size

## Experiment Internals (exp2_benchmarks.py)

### AMM math interface
exp2 uses a local `_v3_output_local(q, sqrt_price_x96, liquidity, fee_bps, direction)` rather
than importing src.amm_math.v3_total_output (incompatible bool vs string interface).
src.amm_math.v2_total_output IS imported directly (compatible signature).

### Bisection for optimal allocation (gas_free_optimal_allocation)
- Marginal price equalization via brentq on excess supply function
- Critical edge case: for very high-liquidity V3 pools (L ~ 5e22) and typical trade
  sizes (Q ~ 1–100 WETH), lam_star is within float64 precision of spot_j, so:
  1. Iteratively narrow lam_high toward max_spot (64 halvings, stops if gap < ε_mach)
  2. If bracket never found → all Q goes to highest-spot pool (correct for L >> Q/P)
  3. Post-brentq: if alloc.sum() ≈ 0 (numerical collapse) → same fallback
- With realistic pool liquidity, all Q typically routes to pool 1 (5bps, highest spot)

### Shortfall conventions
- shortfall_fvo = FVO_raw_output – realized_output  (both in USDC equivalent)
- shortfall_gfvo = GFVO_net_output – realized_output  (GFVO already gas-deducted)
- shortfall_fvo >= shortfall_gfvo always (FVO is the upper bound without gas cost)
- shortfall_bps = shortfall / usdc_equiv_size × 10_000

### G-FVO gas handling
- Enumerate all 15 non-empty subsets of 4 pools
- Q_eff = Q – sum(gas_costs_in_input_tokens[j] for j in S): skip if ≤ 0
- Solve gas-free optimal over S for full Q (not Q_eff)
- net = AMM_output – sum(gas_costs_usdc[j] for j in S)
- GAS_COSTS_USDC derived from config: GAS_COSTS_ETH[pid] × GAS_ETH_PRICE_WEI / 1e18 × price

## Running Experiments
```python
import sys; sys.path.insert(0, '/workspace/project')
from exp.exp1_baseline import run_experiment as exp1; exp1(n_synthetic=50_000)
from exp.exp2_benchmarks import run_experiment as exp2; exp2(n_synthetic=50_000)
from exp.exp3_staleness import run_experiment as exp3; exp3(n_synthetic=50_000)
from exp.exp4_trade_size import run_experiment as exp4; exp4(n_synthetic=50_000)
from exp.exp5_sandwich import run_experiment as exp5; exp5(n_synthetic=50_000)
```
Results saved to /workspace/project/results/*.csv
