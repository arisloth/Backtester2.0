"""
main.py — Entry point for running a backtest.

Configure your backtest here and run:
    python main.py

Edit the CONFIG section below to change symbols, dates, strategy,
slippage/commission models, and analytics options.
"""

import logging
import os

# ------------------------------------------------------------------
# Logging — set to INFO for progress, DEBUG for full event trace
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==================================================================
# CONFIG — edit this section to configure your backtest
# ==================================================================

CONFIG = {
    # --- Data ---
    "data_source": "yfinance",       # "yfinance" | "alpaca" | "ccxt" | "forex"
    "symbols":     ["SPY"],
    "start":       "2020-01-01",
    "end":         "2024-12-31",
    "interval":    "1d",             # yfinance: "1d","1h" etc. | alpaca: "1Day","1Hour"

    # --- Strategy ---
    "strategy":    "sma_cross",      # "sma_cross" | "fvg"
    "fast":        50,               # SMA fast period
    "slow":        200,              # SMA slow period

    # --- FVG strategy params (used when strategy = "fvg") ---
    "fvg_direction":         "long",   # "long" | "short" | "both"
    "fvg_atr_period":        14,
    "fvg_atr_stop_mult":     0.75,     # stop = gap_low - N * ATR
    "fvg_tp_atr_mult":       3.0,      # TP   = fill_price + N * ATR
    "fvg_ema200_filter":     True,
    "fvg_order_block_filter":True,
    "fvg_min_gap_atr":       0.25,
    "fvg_max_gap_age":       10,
    "fvg_max_hold_bars":     50,
    "fvg_tp1_enabled":       True,
    "fvg_tp1_ratio":         1.0,

    # --- Capital & sizing ---
    "initial_capital":    1_000.0,
    "risk_pct":           0.02,      # fraction of current equity to risk per trade (stop-based sizing)
    "position_size_pct":  0.10,      # fallback fraction used when signal has no stop price
    "short_borrow_rate":  0.0,       # annualized borrow cost for short positions (e.g. 0.03 = 3%)

    # --- Slippage model ---
    # "fixed" | "volatility" | "volume_impact"
    "slippage_model": "fixed",
    "slippage_pct":   0.0005,        # used by fixed model (0.05%)
    "atr_multiplier": 0.1,           # used by volatility model
    "impact_factor":  0.1,           # used by volume_impact model

    # --- Commission model ---
    # "zero" | "per_share" | "percent" | "spread"
    "commission_model":   "zero",
    "commission_rate":    0.005,     # per_share: $/share | percent: fraction
    "commission_minimum": 1.0,       # per_share minimum
    "spread_pips":        2.0,       # forex spread model

    # --- Partial fills ---
    "fill_ratio": 1.0,               # 1.0 = full fill, 0.5 = 50% partial

    # --- Analytics ---
    "risk_free_rate":    0.0,
    "periods_per_year":  252,
    "monte_carlo_n":     1000,
    "monte_carlo_dd_threshold": 0.20,

    # --- Output ---
    # Set to a directory path to save charts as PNGs, or None to display interactively
    "chart_output_dir": None,
}

# ==================================================================
# End of CONFIG
# ==================================================================


# Re-export builders so optimizer.py and other modules can continue to
# use `from main import build_*` without changes.
from builders import (  # noqa: E402
    build_data_handler,
    build_strategy,
    build_fill_model,
    build_cost_model,
)


def run(cfg: dict = None) -> dict:
    """
    Run a full backtest from CONFIG (or a custom cfg dict) and return metrics.
    """
    if cfg is None:
        cfg = CONFIG

    from core.portfolio import Portfolio
    from core.broker import Broker
    from core.engine import Engine

    feed       = build_data_handler(cfg)
    strategies = [build_strategy(cfg, s) for s in cfg["symbols"]]
    portfolio  = Portfolio(
        initial_capital=cfg["initial_capital"],
        position_size_pct=cfg["position_size_pct"],
        risk_pct=cfg.get("risk_pct", 0.02),
        short_borrow_rate=cfg.get("short_borrow_rate", 0.0),
    )
    broker = Broker(
        fill_model=build_fill_model(cfg),
        cost_model=build_cost_model(cfg),
        fill_ratio=cfg["fill_ratio"],
    )
    engine = Engine(
        data_handler=feed,
        strategies=strategies,
        portfolio=portfolio,
        broker=broker,
    )

    logger.info(
        f"Starting backtest: {cfg['symbols']} | {cfg['data_source']} | "
        f"{cfg['start']} → {cfg['end']} | strategy={cfg['strategy']}"
    )
    engine.run()

    from analytics.metrics import compute_all, print_summary
    eq     = portfolio.equity_series()
    trades = portfolio.trade_dataframe()
    metrics = compute_all(
        eq, trades,
        risk_free_rate=cfg["risk_free_rate"],
        periods_per_year=cfg["periods_per_year"],
    )
    print_summary(metrics, initial_capital=cfg["initial_capital"])

    if not trades.empty:
        from analytics.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            trades,
            initial_capital=cfg["initial_capital"],
            n=cfg["monte_carlo_n"],
            dd_threshold=cfg["monte_carlo_dd_threshold"],
            return_paths=True,
            seed=42,
        )
        print(mc.summary())
        metrics["monte_carlo"] = mc
    else:
        logger.info("No completed trades — skipping Monte Carlo.")

    from analytics.visualizer import plot_all
    plot_all(eq, trades, save_dir=cfg["chart_output_dir"])

    from results import save_results
    save_results(cfg, metrics, eq, trades)

    return metrics


def run_optimize(opt_cfg: dict) -> None:
    """Run optimizer or walk-forward from a prompt_optimize_config() result."""
    import json
    from datetime import datetime
    from analytics.optimizer import optimize, walk_forward_months
    from results import next_run_number

    mode       = opt_cfg["mode"]
    base_cfg   = opt_cfg["base_cfg"]
    param_grid = opt_cfg["param_grid"]
    metric     = opt_cfg["metric"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_num   = next_run_number()
    run_dir   = os.path.join("results", f"run_{run_num}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    if mode == "simple":
        result = optimize(
            base_cfg=base_cfg,
            param_grid=param_grid,
            is_start=opt_cfg["is_start"],
            is_end=opt_cfg["is_end"],
            oos_start=opt_cfg["oos_start"],
            oos_end=opt_cfg["oos_end"],
            metric=metric,
        )

        print("\n" + "=" * 50)
        print("  Optimization Result")
        print("=" * 50)
        print(f"  Best params:   {result.best_params}")
        print(f"  IS  {metric}: {result.best_is_metrics.get(metric, 'n/a'):.4f}")
        print(f"  OOS {metric}: {result.oos_metrics.get(metric, 'n/a'):.4f}")
        print(f"  OOS trades:    {result.oos_metrics.get('total_trades', 0)}")
        print(f"  OOS win rate:  {result.oos_metrics.get('win_rate', 0):.1%}")
        display_cols = list(param_grid.keys()) + [metric, "total_trades"]
        display_cols = [c for c in display_cols if c in result.all_results.columns]
        print(f"\n  Top IS combinations:")
        print(result.all_results[display_cols].head(10).to_string(index=False))

        payload = {
            "mode": "simple",
            "best_params": result.best_params,
            "is_metrics":  {k: v for k, v in result.best_is_metrics.items()
                            if isinstance(v, (int, float, str, bool, type(None)))},
            "oos_metrics": {k: v for k, v in result.oos_metrics.items()
                            if isinstance(v, (int, float, str, bool, type(None)))},
            "is_start":  opt_cfg["is_start"],  "is_end":  opt_cfg["is_end"],
            "oos_start": opt_cfg["oos_start"], "oos_end": opt_cfg["oos_end"],
            "metric": metric,
            "_config": {k: v for k, v in base_cfg.items()
                        if isinstance(v, (int, float, str, bool, list, type(None)))},
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
        result.all_results.to_csv(os.path.join(run_dir, "all_runs.csv"), index=False)

        from analytics.report import optimize_report, save_report
        report_text = optimize_report(base_cfg, result, opt_cfg)
        print(report_text)
        save_report(report_text, run_dir)
        print(f"\n  Results saved to: {run_dir}/")
        print(f"    summary.json  |  all_runs.csv  |  report.txt")

    else:  # walkforward
        result = walk_forward_months(
            base_cfg=base_cfg,
            param_grid=param_grid,
            start=opt_cfg["wf_start"],
            end=opt_cfg["wf_end"],
            train_months=opt_cfg["train_months"],
            test_months=opt_cfg["test_months"],
            metric=metric,
            n_jobs=opt_cfg.get("n_jobs", 1),
        )

        print("\n" + "=" * 50)
        print("  Walk-Forward Result")
        print("=" * 50)
        print(f"  OOS Sharpe (weighted): {result.oos_sharpe:.4f}")
        print(f"  OOS Win Rate:          {result.oos_win_rate:.1%}")
        print(f"  Total OOS trades:      {result.oos_total_trades}")
        print(f"\n  Per-window summary:")
        print(result.summary.to_string(index=False))

        result.summary.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump({
                "mode":             "walkforward",
                "oos_sharpe":       result.oos_sharpe,
                "oos_win_rate":     result.oos_win_rate,
                "oos_total_trades": result.oos_total_trades,
                "train_months":     opt_cfg["train_months"],
                "test_months":      opt_cfg["test_months"],
                "start": opt_cfg["wf_start"], "end": opt_cfg["wf_end"],
                "metric": metric,
                "_config": {k: v for k, v in base_cfg.items()
                            if isinstance(v, (int, float, str, bool, list, type(None)))},
            }, f, indent=2, default=str)

        import pandas as pd
        fold_trades = [w.oos_trades for w in result.windows
                       if w.oos_trades is not None and not w.oos_trades.empty]
        all_oos_trades = pd.concat(fold_trades, ignore_index=True) if fold_trades else pd.DataFrame()
        if not all_oos_trades.empty:
            all_oos_trades.to_csv(os.path.join(run_dir, "trades.csv"), index=False)

        from analytics.report import walkforward_report, save_report
        report_text = walkforward_report(base_cfg, result, opt_cfg, trades=all_oos_trades)
        print(report_text)
        save_report(report_text, run_dir)

        saved_files = "summary.csv  |  summary.json  |  report.txt"
        if not all_oos_trades.empty:
            saved_files += "  |  trades.csv"
        print(f"\n  Results saved to: {run_dir}/")
        print(f"    {saved_files}")


if __name__ == "__main__":
    import sys
    from cli import prompt_config, prompt_optimize_config

    if "--no-prompt" in sys.argv or "-y" in sys.argv:
        run()
    else:
        from cli import _choose as choose
        print("\n" + "=" * 50)
        print("  Backtester")
        print("=" * 50)
        mode = choose("What would you like to do?", [
            ("Backtest",        "backtest"),
            ("Optimize IS/OOS", "optimize"),
            ("Walk-Forward",    "walkforward"),
        ], default_index=0)

        if mode == "backtest":
            cfg = prompt_config()
            if cfg:
                run(cfg)
        else:
            opt_cfg = prompt_optimize_config()
            if opt_cfg:
                run_optimize(opt_cfg)
