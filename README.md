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
from Polymarket's public price history): mean pinball loss **0.304230** vs
v2's 0.310602, v1's 0.323229 and the house baseline's 0.363589 — all six
cases improve on v2 (p00 0.4735, p01 0.3112, p02 0.4089, p03 0.3354,
p04 0.2490, p05 0.1072). Verified by running this main.ts on the practice
inputs (same score, two runs byte-identical, ~1 s per case). What's new in
v3: (1) volatility conditioning — quantile widths scaled 0.6/1.0/1.3 by
recent realized vol vs its 24h hourly rate, only for markets with >= 480
minutes of history; (2) cross-market sum drift — the median drifts toward
the 24-market sum's target of 1; (3) post-up-jump median drift — after a
1-minute rise >= 0.02 the median alone shifts down (tails stay put).
Horizon-decay of widths and lower-half jump drifts were tested and rejected.

Run: `node main.ts` (Node 22, stdlib only). Reads `$MAKEFUN_INPUT_DIR`
(default `/input`), writes `$MAKEFUN_OUTPUT_DIR` (default `/output`).
