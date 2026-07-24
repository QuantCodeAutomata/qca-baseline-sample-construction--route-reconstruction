# QCA Baseline — WETH–USDC Same-Pair Routing Analysis

Replication repository for a five-experiment study on execution quality and
routing inefficiency in the WETH–USDC Uniswap pool universe on Ethereum mainnet.

---

## Overview

The paper analyses same-pair swaps across exactly four Ethereum L1 WETH–USDC
pools:

| Pool | Protocol | Fee |
|---|---|---|
| V2 30 bps | Uniswap V2 | 0.30 % |
| V3 1 bp | Uniswap V3 | 0.01 % |
| V3 5 bps | Uniswap V3 | 0.05 % |
| V3 30 bps | Uniswap V3 | 0.30 % |

Sample period: Ethereum mainnet blocks **19,500,000–23,000,000**.

Because a full Ethereum archive node is required to access real blockchain data,
all experiments default to a synthetic data generator (`src/synthetic_data.py`)
calibrated to reproduce the paper's reported concentration moments and
trade-size distribution.

---

## Repository Structure

```
.
├── src/                        # Core implementations
│   ├── config.py               # Pool universe, addresses, block range
│   ├── amm_math.py             # Exact Uniswap V2 and V3 output functions
│   ├── concentration.py        # Activation rate, entropy, HHI
│   ├── filters.py              # Counterflow and min-USDC filters
│   ├── router_labels.py        # Address → router label mapping
│   └── synthetic_data.py       # Calibrated synthetic data generator
│
├── exp/                        # Experiment scripts (run as modules)
│   ├── exp1_baseline.py        # Sample construction and concentration metrics
│   ├── exp2_benchmarks.py      # SCO, FVO, G-FVO optimal routing benchmarks
│   ├── exp3_staleness.py       # State-staleness sensitivity analysis
│   ├── exp4_trade_size.py      # Trade-size heterogeneity and tail analysis
│   └── exp5_sandwich.py        # Sandwich-attack subsample analysis
│
├── tests/                      # 164 pytest tests (all passing)
│   ├── test_amm_math.py
│   ├── test_benchmarks.py
│   ├── test_concentration.py
│   ├── test_filters.py
│   ├── test_integration.py
│   ├── test_sandwich.py
│   ├── test_staleness.py
│   └── test_trade_size.py
│
├── data/
│   ├── raw/                    # Raw blockchain logs (not committed)
│   ├── interim/                # Processed parquet files (not committed)
│   └── fixtures/               # Small deterministic test fixtures
│
├── results/
│   ├── RESULTS.md              # Full experiment results summary
│   ├── exp1_summary.csv        # Exp 1 sample statistics
│   ├── exp1_router_stats.csv   # Exp 1 router-level breakdown
│   ├── exp2_summary.csv        # Exp 2 aggregate benchmark shortfalls
│   ├── exp2_shortfalls.csv     # Exp 2 row-level benchmark data (50 K rows)
│   ├── exp3_staleness.csv      # Exp 3 per-transaction staleness data
│   ├── exp3_staleness_per_n.csv# Exp 3 per-N aggregate
│   ├── exp4_size_bins.csv      # Exp 4 trade-size bin statistics
│   └── exp5_sandwich.csv       # Exp 5 sandwich comparison table
│
├── requirements.txt
├── .gitignore
└── AGENTS.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all five experiments
PYTHONPATH=. python exp/exp1_baseline.py
PYTHONPATH=. python exp/exp2_benchmarks.py
PYTHONPATH=. python exp/exp3_staleness.py
PYTHONPATH=. python exp/exp4_trade_size.py
PYTHONPATH=. python exp/exp5_sandwich.py

# Run full test suite (164 tests)
pytest
```

---

## Experiments

### Exp 1 — Baseline Sample and Concentration Metrics

Constructs the filtered transaction-level dataset.  Applies the paper's two
filters (remove counterflow transactions; drop USDC-side amount < 100) and
computes per-transaction concentration metrics:

- **Activation rate** κ = |{j : x_j > 0}| / 4
- **Entropy** H = −Σ x_j log x_j  (0 log 0 = 0)
- **HHI** = Σ x_j²

Produces `results/exp1_summary.csv` and `results/exp1_router_stats.csv`.

### Exp 2 — SCO, FVO, and G-FVO Benchmarks

Computes three execution-time optimal routing benchmarks for each transaction:

| Benchmark | Optimisation domain |
|---|---|
| **SCO** | Realised activated set only |
| **FVO** | Full four-pool universe (no gas) |
| **G-FVO** | Best subset from 15 non-empty subsets with fixed activation gas |

Gas-free allocation uses marginal-price equalisation solved via bisection.
G-FVO uses exhaustive subset enumeration over 2⁴−1 = 15 candidates.

Produces `results/exp2_summary.csv` and `results/exp2_shortfalls.csv`.

### Exp 3 — State-Staleness Analysis

Compares FVO and G-FVO performance when routing decisions use pool states
from N ∈ {0, 1, …, 10} blocks before execution.  Documents the N = 0 → N = 1
performance jump.

Produces `results/exp3_staleness.csv` and `results/exp3_staleness_per_n.csv`.

### Exp 4 — Trade-Size Heterogeneity

Bins transactions by USDC-equivalent size ([100,1K), [1K,10K), [10K,100K),
[100K,∞)) and reports mean/median/quartile shortfalls in basis points and
USD terms.  Quantifies heavy-tail concentration of dollar losses.

Produces `results/exp4_size_bins.csv`.

### Exp 5 — Sandwich-Attack Subsample

Tags transactions sandwiched within the same block by opposite-direction swaps
from the same counterparty cluster.  Compares SCO, FVO, and G-FVO shortfalls
for sandwiched vs. non-sandwiched subsets.

Produces `results/exp5_sandwich.csv`.

---

## Core Algorithms

### Uniswap V2 Output (`src/amm_math.py`)

```
Δy = (fee_num · Δx · y) / (fee_denom · x + fee_num · Δx)
```

where `fee_num = 10000 − fee_bps` and `fee_denom = 10000`.

### Uniswap V3 Output (`src/amm_math.py`)

Single-range constant-L approximation with exact sqrt-price arithmetic:

```
Δ√P = Δx / L          (token-in = token-0)
Δ√P = Δy / L          (token-in = token-1)
```

Fee is applied before computing output.

### Gas-Free Optimal Routing (`exp/exp2_benchmarks.py`)

KKT optimality: active pools share a common marginal price λ*.

Bisection solves S(λ) = Σⱼ mp_j⁻¹(λ) = Q, where mp_j⁻¹(λ) is the
input that sets marginal output rate of pool j to λ.

---

## Using Real Blockchain Data

Set `ETH_RPC_URL` to a JSON-RPC archive endpoint (e.g. Infura, Alchemy,
or a self-hosted Erigon/Reth node) and replace `USE_SYNTHETIC = True` in
`src/config.py` with `USE_SYNTHETIC = False`.  The data loader skeleton
in `src/config.py` documents the required log schema.

Event topics to decode:

- **V2 Sync**: `0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1`
- **V3 Swap**: `0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67`

---

## Test Suite Summary

```
164 passed  ·  0 failed  ·  0 errors
```

Key property tests:
- V2/V3 output ≥ 0 for any non-negative input
- Activation rate ∈ [0, 1]; entropy ≥ 0; HHI ∈ (0, 1]
- Single-pool routes: κ = 0.25, H = 0, HHI = 1
- FVO shortfall ≥ SCO shortfall for every transaction
- All transactions assigned exactly one size bin
- Sandwich bracketing logic correct (victim flagged, non-victim clean)
