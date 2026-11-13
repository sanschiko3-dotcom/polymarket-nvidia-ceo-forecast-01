# Polymarket NVIDIA CEO forecaster

Forecasts the Yes price of every trading market of Polymarket's "Next CEO of
NVIDIA?" event, minute by minute, 4 hours ahead, as five quantiles per minute.

Approach: per-market **empirical quantiles of past h-step price changes** from
the last 24 hours of minute prices, shrunk 70% toward zero width, interpolated
in log horizon. Prices sit still most minutes and then jump asymmetrically, so
narrow-centred, fat-tailed empirical quantiles beat a normal random walk.

Practice backtest (6 cases, ground truth from the next case's prices; p05's
from Polymarket's public price history): mean pinball loss **0.323229** vs the
house baseline's 0.363589.

Run: `node main.ts` (Node 22, stdlib only). Reads `$MAKEFUN_INPUT_DIR`
(default `/input`), writes `$MAKEFUN_OUTPUT_DIR` (default `/output`).
