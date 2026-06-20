"""
api/config_schema.py — metadata that drives the dashboard's New Run form.

Defaults come straight from main.CONFIG (single source of truth), so adding a
config key there makes it appear in the form automatically. This module only
layers on the things CONFIG can't express: which select-fields are enums (and
their choices), how fields group into sections, and which sections are
strategy-specific (shown conditionally on the chosen strategy).
"""

from typing import Optional

# Enumerated fields → allowed values (everything else is inferred from its default).
ENUMS = {
    "data_source":        ["yfinance", "alpaca", "ccxt", "forex"],
    "strategy":           ["sma_cross", "fvg", "ema_pullback"],
    "slippage_model":     ["fixed", "volatility", "volume_impact"],
    "commission_model":   ["zero", "per_share", "percent", "spread"],
    "monte_carlo_method": ["iid", "block"],
    "fvg_direction":      ["long", "short", "both"],
    "ep_direction":       ["long", "short", "both"],
    "ep_pullback_ema":    ["ema_fast", "ema_slow"],
    "ep_runner_mode":     ["structure", "atr_trail", "fixed_r"],
    "ep_btc_gate_mode":   ["ema_stack", "ema20_reclaim", "off"],
    "ep_rs_filter_sides": ["short", "both", "off"],
}

# Valid timeframes per data source (from CLAUDE.md).
INTERVALS = {
    "yfinance": ["1m", "5m", "15m", "1h", "4h", "1d", "1wk"],
    "alpaca":   ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
    "ccxt":     ["1m", "5m", "15m", "1h", "4h", "12h", "1d", "3d", "1w"],
    "forex":    ["1m", "5m", "15m", "1h", "4h", "1d", "1wk"],
}

# Metric a grid search can maximize on the in-sample window.
OPTIMIZER_METRICS = [
    "sharpe_ratio", "sortino_ratio", "cagr", "expectancy", "profit_factor",
]

# Section layout: (key, label, [config keys], optional strategy tag). ep_* keys
# are expanded dynamically so the whole EMA-pullback param set stays in sync.
_SECTIONS = [
    ("data", "Data", ["data_source", "symbols", "start", "end", "interval",
                      "cache_ttl_days", "refresh_cache"], None),
    ("strategy_sma", "Strategy — SMA Crossover", ["fast", "slow"], "sma_cross"),
    ("strategy_fvg", "Strategy — Fair Value Gap",
        ["fvg_direction", "fvg_atr_period", "fvg_atr_stop_mult", "fvg_tp_atr_mult",
         "fvg_ema200_filter", "fvg_order_block_filter", "fvg_min_gap_atr",
         "fvg_max_gap_age"], "fvg"),
    ("strategy_ep", "Strategy — EMA Pullback", "EP_DYNAMIC", "ema_pullback"),
    ("capital", "Capital & Sizing",
        ["initial_capital", "risk_pct", "position_size_pct", "short_borrow_rate",
         "short_initial_margin"], None),
    ("slippage", "Slippage",
        ["slippage_model", "slippage_pct", "atr_multiplier", "impact_factor"], None),
    ("commission", "Commission",
        ["commission_model", "commission_rate", "commission_minimum", "spread_pips"], None),
    ("fills", "Fills", ["fill_ratio", "min_fill_volume"], None),
    ("analytics", "Analytics",
        ["risk_free_rate", "periods_per_year", "monte_carlo_n",
         "monte_carlo_dd_threshold", "monte_carlo_method", "monte_carlo_block_size"], None),
]


def _label(key: str) -> str:
    """Prettify a config key into a form label."""
    pretty = key.replace("ep_", "").replace("fvg_", "").replace("_", " ").strip()
    return pretty[:1].upper() + pretty[1:]


def _field_type(key: str, default) -> str:
    if key in ENUMS:
        return "select"
    if isinstance(default, bool):
        return "boolean"
    if isinstance(default, int):
        return "integer"
    if isinstance(default, float):
        return "number"
    if isinstance(default, list):
        return "list"
    return "text"  # str or None


def _field(key: str, cfg: dict) -> dict:
    default = cfg.get(key)
    f = {"key": key, "label": _label(key), "type": _field_type(key, default), "default": default}
    if key in ENUMS:
        f["options"] = ENUMS[key]
    return f


def build_schema() -> dict:
    """Return the full form schema: options, sections, run types, and defaults."""
    from main import CONFIG

    ep_keys = [k for k in CONFIG if k.startswith("ep_")]

    sections = []
    for skey, label, keys, strat in _SECTIONS:
        if keys == "EP_DYNAMIC":
            keys = ep_keys
        fields = [_field(k, CONFIG) for k in keys if k in CONFIG]
        section = {"key": skey, "label": label, "fields": fields}
        if strat is not None:
            section["strategy"] = strat
        sections.append(section)

    defaults = {
        k: v for k, v in CONFIG.items()
        if isinstance(v, (int, float, str, bool, list, type(None)))
    }

    return {
        "run_types": ["backtest", "optimize", "walkforward"],
        "options": {
            **{k: v for k, v in ENUMS.items()},
            "intervals": INTERVALS,
            "optimizer_metrics": OPTIMIZER_METRICS,
        },
        "sections": sections,
        "defaults": defaults,
    }
