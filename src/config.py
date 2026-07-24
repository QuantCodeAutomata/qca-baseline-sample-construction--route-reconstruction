"""
Project-wide configuration: pool universe, router labels, token addresses,
block range, and gas cost calibration.
"""

POOL_UNIVERSE: dict[int, dict] = {
    0: {
        "address": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
        "protocol": "v2",
        "fee_bps": 30,
        "name": "Uniswap v2 30bps",
    },
    1: {
        "address": "0xE0554a476A092703abdB3Ef35c80e0D76d32939F",
        "protocol": "v3",
        "fee_bps": 1,
        "name": "Uniswap v3 1bp",
    },
    2: {
        "address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        "protocol": "v3",
        "fee_bps": 5,
        "name": "Uniswap v3 5bps",
    },
    3: {
        "address": "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",
        "protocol": "v3",
        "fee_bps": 30,
        "name": "Uniswap v3 30bps",
    },
}

# Maps lowercase router address → human-readable label.
ROUTER_LABELS: dict[str, str] = {
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "Universal Router",
    "0x9008d19f58aabd9ed0d60971565aa8510560ab41": "CoWSwap",
    "0x1111111254fb6c44bac0bed2854e76f90643097d": "1inch v4",
    "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch v5",
    "0xca423977156bb05b13a2ba3b76bc5419e2fe9680": "Odos v2",
}

WETH_ADDRESS: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC_ADDRESS: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

WETH_DECIMALS: int = 18
USDC_DECIMALS: int = 6

BLOCK_START: int = 19_500_000
BLOCK_END: int = 23_000_000

NUM_POOLS: int = 4

MIN_USDC_SIZE: float = 100.0  # minimum USDC-equivalent trade size

# Gas cost per pool (in gas units) for the G-FVO fixed calibration.
GAS_COSTS_ETH: dict[int, int] = {
    0: 50_000,
    1: 80_000,
    2: 80_000,
    3: 80_000,
}

GAS_ETH_PRICE_WEI: int = 20 * 10**9  # 20 gwei default
