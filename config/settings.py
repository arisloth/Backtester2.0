"""
config/settings.py — API keys and global defaults.

Set your keys here or via environment variables. Never commit real keys.
"""

import os

# ------------------------------------------------------------------
# Alpaca
# ------------------------------------------------------------------
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
# Alpaca uses ALPACA_SECRET_KEY in their own CLI tools — support both names
ALPACA_API_SECRET = os.getenv("ALPACA_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))

# "iex" = free delayed feed, "sip" = paid full market feed
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex")

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_INITIAL_CAPITAL   = 100_000.0
DEFAULT_POSITION_SIZE_PCT = 0.10
DEFAULT_SLIPPAGE_PCT      = 0.0005   # 0.05% fixed slippage
DEFAULT_RISK_FREE_RATE    = 0.0
DEFAULT_PERIODS_PER_YEAR  = 252      # daily bars
