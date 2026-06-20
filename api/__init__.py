"""
api/ — programmatic + HTTP surface around the backtester.

api.runner        headless entry points (run a backtest / optimize / walk-forward,
                  persist to the DB, return the new run id).
api.config_schema form metadata + defaults that drive the dashboard's New Run UI.

The FastAPI app (added in Phase 3) builds on these.
"""
