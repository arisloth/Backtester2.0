"""
data/alignment.py -- Shared multi-symbol timestamp alignment helpers.
"""

import logging
from typing import Dict

import pandas as pd


def align_symbol_data(
    data: Dict[str, pd.DataFrame],
    logger: logging.Logger,
    context: str = "Symbols",
) -> Dict[str, pd.DataFrame]:
    """
    Inner-join symbol DataFrames on timestamp and log data-loss diagnostics.

    The returned DataFrames preserve existing feed behavior: only timestamps
    present for every symbol survive, and rows are reset to integer indexes.
    """
    timestamp_sets = {
        symbol: set(df["timestamp"])
        for symbol, df in data.items()
    }
    common_ts = set.intersection(*timestamp_sets.values()) if timestamp_sets else set()
    union_ts = set.union(*timestamp_sets.values()) if timestamp_sets else set()

    if not common_ts:
        raise ValueError("No overlapping timestamps across symbols after alignment.")

    aligned_len = len(common_ts)
    total_removed = len(union_ts) - aligned_len
    log = logger.warning if total_removed else logger.info
    log(
        "%s aligned to %s common bars; %s/%s union timestamps removed by inner join.",
        context,
        aligned_len,
        total_removed,
        len(union_ts),
    )

    aligned = {}
    for symbol, df in data.items():
        original_len = len(df)
        dropped = original_len - aligned_len
        missing_vs_union = len(union_ts - timestamp_sets[symbol])
        symbol_log = logger.warning if dropped or missing_vs_union else logger.info
        symbol_log(
            "%s alignment %s: original=%s aligned=%s dropped=%s missing_vs_union=%s",
            context,
            symbol,
            original_len,
            aligned_len,
            dropped,
            missing_vs_union,
        )
        aligned[symbol] = (
            df[df["timestamp"].isin(common_ts)]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    return aligned
