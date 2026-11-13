# Polymarket NVIDIA CEO forecaster

Forecasts the Yes price of every trading market of Polymarket's "Next CEO of
NVIDIA?" event, minute by minute, 4 hours ahead, as five quantiles per minute.

Approach: per-market **empirical quantiles of past h-step price changes** from
the last 24 hours of minute prices, with **per-tau asymmetric shrinkage**:
SHRINK = [0.35, 0.80, 0.85, 0.90, 0.70] for tau = 0.05/0.25/0.50/0.75/0.95.
The lower tail is kept wide (big falls are persistent) and the upper tail is
shrunk hard (rises revert); the middle is shrunk hardest (most mass sits at
zero change).

Practice backtest (6 cases, ground truth from the next case's prices; p05's
from Polymarket's public price history): mean pinball loss **0.310602** vs
v1's 0.323229 and the house baseline's 0.363589 — all six cases improve
(p00 0.4818, p01 0.3148, p02 0.4235, p03 0.3436, p04 0.2498, p05 0.1117).
Verified by running this main.ts on the practice inputs (same score, two
runs byte-identical, ~0.6 s per case). A 24h/72h window mix and an explicit
post-up-jump drift were tested and rejected (both worse than v2).

Run: `node main.ts` (Node 22, stdlib only). Reads `$MAKEFUN_INPUT_DIR`
(default `/input`), writes `$MAKEFUN_OUTPUT_DIR` (default `/output`).
