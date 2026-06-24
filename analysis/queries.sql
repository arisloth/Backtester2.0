-- analysis/queries.sql
-- Analytical SQL over the backtester results database (backtester.db).
--
-- The schema is a normalized star around `runs`:
--   runs 1─*  trades            (per round-trip trade)
--   runs 1─1  metrics           (headline performance per run)
--   runs 1─*  equity_points     (mark-to-market equity curve)
--   runs 1─*  wf_windows        (walk-forward rolling windows: IS vs OOS)
--   runs 1─1  optimizer_results 1─* optimizer_trials (grid-search leaderboard)
--
-- Run interactively with:  sqlite3 backtester.db ".read analysis/queries.sql"
-- or load individual queries from the companion notebook (results_analysis.ipynb).


-- ---------------------------------------------------------------------------
-- 1. Run inventory — what's in the database, by type / strategy / source.
-- ---------------------------------------------------------------------------
SELECT run_type,
       strategy,
       data_source,
       COUNT(*) AS runs
FROM runs
GROUP BY run_type, strategy, data_source
ORDER BY run_type, runs DESC;


-- ---------------------------------------------------------------------------
-- 2. Strategy scorecard — join runs to their headline metrics and compare
--    strategies on risk-adjusted return, hit rate, and drawdown.
--    (JOIN + GROUP BY — the core analyst comparison.)
-- ---------------------------------------------------------------------------
SELECT r.strategy,
       COUNT(*)                          AS runs,
       ROUND(AVG(m.sharpe_ratio), 2)     AS avg_sharpe,
       ROUND(AVG(m.sortino_ratio), 2)    AS avg_sortino,
       ROUND(AVG(m.win_rate), 3)         AS avg_win_rate,
       ROUND(AVG(m.profit_factor), 2)    AS avg_profit_factor,
       ROUND(AVG(m.max_drawdown_pct), 3) AS avg_max_drawdown
FROM runs r
JOIN metrics m ON m.run_id = r.id
WHERE r.run_type = 'backtest'
GROUP BY r.strategy
ORDER BY avg_sharpe DESC;


-- ---------------------------------------------------------------------------
-- 3. Per-symbol, per-side profitability — which instruments and directions
--    actually carry the edge. (GROUP BY with conditional aggregation.)
-- ---------------------------------------------------------------------------
SELECT symbol,
       side,
       COUNT(*)                                              AS trades,
       ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_pct,
       ROUND(AVG(pnl_pct) * 100, 3)                          AS avg_return_pct,
       ROUND(SUM(pnl), 0)                                    AS total_pnl
FROM trades
GROUP BY symbol, side
HAVING COUNT(*) >= 20
ORDER BY total_pnl DESC;


-- ---------------------------------------------------------------------------
-- 4. Exit-reason economics — how trades close and what each exit type
--    contributes. Surfaces the classic asymmetry: many small stops vs a few
--    larger take-profits. (Conditional aggregation.)
-- ---------------------------------------------------------------------------
SELECT exit_reason,
       COUNT(*)                                              AS trades,
       ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_pct,
       ROUND(AVG(pnl_pct) * 100, 3)                          AS avg_return_pct,
       ROUND(SUM(pnl), 0)                                    AS total_pnl
FROM trades
GROUP BY exit_reason
ORDER BY total_pnl DESC;


-- ---------------------------------------------------------------------------
-- 5. Hold-time vs return — does holding longer help or hurt?
--    (CASE bucketing of a continuous variable.)
-- ---------------------------------------------------------------------------
SELECT CASE
         WHEN hold_bars <  10 THEN '1. <10 bars'
         WHEN hold_bars <  30 THEN '2. 10-29'
         WHEN hold_bars <  60 THEN '3. 30-59'
         WHEN hold_bars < 120 THEN '4. 60-119'
         ELSE                      '5. 120+'
       END                            AS hold_bucket,
       COUNT(*)                       AS trades,
       ROUND(AVG(pnl_pct) * 100, 3)   AS avg_return_pct,
       ROUND(AVG(hold_bars), 1)       AS avg_hold_bars
FROM trades
GROUP BY hold_bucket
ORDER BY hold_bucket;


-- ---------------------------------------------------------------------------
-- 6. Overfitting check — average in-sample vs out-of-sample Sharpe across all
--    walk-forward windows. A large IS→OOS drop signals curve-fitting.
-- ---------------------------------------------------------------------------
SELECT COUNT(*)                                  AS windows,
       ROUND(AVG(is_sharpe), 2)                  AS avg_is_sharpe,
       ROUND(AVG(oos_sharpe), 2)                 AS avg_oos_sharpe,
       ROUND(AVG(is_sharpe) - AVG(oos_sharpe), 2) AS sharpe_degradation,
       ROUND(100.0 * SUM(CASE WHEN oos_sharpe > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_oos_positive
FROM wf_windows;


-- ---------------------------------------------------------------------------
-- 7. Equity curve + drawdown, computed in SQL with a window function.
--    Running peak via MAX(...) OVER (...) lets us derive drawdown without
--    pulling the whole series into pandas first. Swap :run_id as needed.
-- ---------------------------------------------------------------------------
WITH curve AS (
    SELECT timestamp,
           equity,
           MAX(equity) OVER (ORDER BY timestamp ROWS UNBOUNDED PRECEDING) AS running_peak
    FROM equity_points
    WHERE run_id = (SELECT run_id FROM equity_points
                    GROUP BY run_id ORDER BY COUNT(*) DESC LIMIT 1)
)
SELECT timestamp,
       equity,
       running_peak,
       ROUND(100.0 * (equity / running_peak - 1.0), 2) AS drawdown_pct
FROM curve
ORDER BY timestamp;
