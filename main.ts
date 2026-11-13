// polymarket-nvidia-ceo-forecast-01 entry v1 (Node 22, standard library only).
//
// Forecaster: per-market EMPIRICAL quantiles of past h-step price changes,
// shrunk 70% toward zero width.
//
// For each market, take its last 24 hours of minute prices (1,440 points) and,
// for a log-spaced grid of horizons h, collect every overlapping h-minute price
// change, take its 5% / 25% / 50% / 75% / 95% quantiles, keep only 30% of each
// (SHRINK = 0.7), and interpolate in log h. The forecast for minute at+h is
//   q_tau(h) = clamp(last price + shrunk empirical quantile_tau(h), 0, 1).
//
// Why this shape, not the house baseline's normal random walk:
// Polymarket Yes prices sit still most minutes (88% of 1-minute moves are
// exactly zero) and then jump, sometimes by 0.4 in a minute, and the jumps are
// asymmetric (big falls are far more common than big rises in this event's
// history). A normal with a volatility matched to the return standard deviation
// is far too wide in the middle and pays pinball loss at every quiet minute.
// The empirical quantiles are narrow in the centre (most mass at zero change)
// and fat in the tails, which is what the data actually looks like — but the
// raw empirical quantiles still overstate the conditional dispersion (the
// history window contains regime shifts, and overlapping samples double-count
// jumps), so shrinking them toward zero helps: on the six practice cases the
// loss falls from 0.3524 at shrink 0.2 to 0.3232 at shrink 0.7, with a broad
// flat minimum at 0.6-0.8 (0.3250 / 0.3232 / 0.3266), while pure persistence
// (shrink 1.0) is much worse at 0.3838 — so some width is genuinely needed.
// Backtest on the six practice cases (ground truth: next case's prices, p05's
// fetched from Polymarket's public price history): mean pinball loss 0.3232
// against the house baseline's 0.3636.
//
// Deterministic: no randomness, no network, no clock reads; two runs on the
// same case write the same bytes.
//
// Input/output: reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes
// <id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as {"forecasts":[{"id","q"}]}
// with exactly horizon_minutes rows of five non-decreasing numbers in [0,1].

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Log-spaced horizons (minutes) at which empirical quantiles are measured. */
const HGRID = [1, 2, 3, 5, 8, 12, 20, 30, 45, 70, 105, 160, 240];
/** Minutes of history used per market (24 hours). */
const HIST_WIN = 1440;
/** Shrinkage of the empirical quantiles toward zero width (0.7 = keep 30%). */
const SHRINK = 0.7;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Linear-interpolation quantile of a sorted array. */
function quantileSorted(s: number[], tau: number): number {
  const n = s.length;
  if (n === 0) return 0;
  if (n === 1) return s[0] as number;
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const f = pos - lo;
  return (s[lo] as number) * (1 - f) + (s[hi] as number) * f;
}

/** Empirical quantiles of h-step changes for each horizon in HGRID. */
function empQuantiles(prices: number[]): number[][] {
  const p = prices.length > HIST_WIN ? prices.slice(prices.length - HIST_WIN) : prices;
  const n = p.length;
  const out: number[][] = [];
  for (const h of HGRID) {
    if (n - h < 20) {
      out.push([0, 0, 0, 0, 0]);
      continue;
    }
    const ch: number[] = new Array(n - h);
    for (let i = 0; i < n - h; i++) ch[i] = (p[i + h] as number) - (p[i] as number);
    ch.sort((a, b) => a - b);
    out.push(QUANTILES.map((t) => quantileSorted(ch, t) * (1 - SHRINK)));
  }
  return out;
}

/** Empirical quantile row for horizon h, interpolated in log h. */
function interpQ(eq: number[][], h: number): number[] {
  if (h <= HGRID[0]) return eq[0] as number[];
  if (h >= HGRID[HGRID.length - 1]) return eq[eq.length - 1] as number[];
  for (let i = 0; i < HGRID.length - 1; i++) {
    const a = HGRID[i] as number;
    const b = HGRID[i + 1] as number;
    if (h >= a && h <= b) {
      const w = (Math.log(h) - Math.log(a)) / (Math.log(b) - Math.log(a));
      const ra = eq[i] as number[];
      const rb = eq[i + 1] as number[];
      return ra.map((v, j) => v * (1 - w) + (rb[j] as number) * w);
    }
  }
  return eq[eq.length - 1] as number[];
}

/** The empirical-quantile forecast of one market: horizon rows of five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const eq = empQuantiles(m.prices);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const q = interpQ(eq, h);
    const row = q.map((v) => price6(x + v));
    for (let j = 1; j < 5; j++) if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
