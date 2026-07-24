"""
Router labelling utilities.

Maps Ethereum ``to_address`` values to human-readable DEX aggregator names
using the canonical address set defined in ``config.ROUTER_LABELS``.
"""

from typing import Optional

import pandas as pd

from config import ROUTER_LABELS


# Re-export for direct import by other modules.
ROUTER_LABEL_MAP: dict[str, str] = ROUTER_LABELS


def get_router_label(to_address: str) -> Optional[str]:
    """Return the human-readable label for a known router address, or None.

    The lookup is case-insensitive: the input is lower-cased before the
    map is consulted.

    Args:
        to_address: Ethereum hex address string (with or without '0x' prefix,
                    any casing).

    Returns:
        Router label string (e.g. ``'Universal Router'``) if the address is a
        known aggregator, otherwise ``None``.
    """
    return ROUTER_LABEL_MAP.get(to_address.lower())


def label_transactions(
    df: pd.DataFrame,
    address_col: str = "to_address",
) -> pd.DataFrame:
    """Add a ``router_label`` column to a transaction DataFrame.

    Each row receives the label of the router it interacted with, or
    ``None`` if the ``to_address`` is not a recognised aggregator.

    Args:
        df:          DataFrame containing at least the column named by
                     ``address_col``.
        address_col: Name of the column holding the recipient (router)
                     address.  Defaults to ``'to_address'``.

    Returns:
        A copy of ``df`` with an additional ``'router_label'`` column.
        The original DataFrame is not modified.

    Raises:
        KeyError: If ``address_col`` is not present in ``df``.
    """
    if address_col not in df.columns:
        raise KeyError(
            f"Column '{address_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out["router_label"] = out[address_col].apply(get_router_label)
    return out
