// polymarket-nvidia-ceo-forecast-01 entry v3 (Node 22, standard library only).
//
// Forecaster: v2's per-market EMPIRICAL quantiles of past h-step price changes
// (24h window, log-h interpolation, per-tau asymmetric shrinkage toward zero
// width, SHRINK = [0.35, 0.80, 0.85, 0.90, 0.70] for tau = 0.05, 0.25, 0.50,
// 0.75, 0.95) PLUS three genuinely different, separately validated additions:
//
// 1. VOLATILITY CONDITIONING. Recent activity strongly predicts near-term
//    activity: pooled over the six practice cases, a market whose last-60-min
//    realized absolute move is under half its 24h hourly rate moves only
//    0.067 in total over the next 240 minutes, versus 0.28-0.37 for markets
//    at/above their hourly rate (correlation +0.40, n=128 market-cases).
//    So each market's quantile widths are scaled by g(rho), rho = (realized
//    |move| over the last 60 min) / (mean hourly realized |move| over the
//    last 24h): rho < 0.5 -> 0.6, rho < 1 -> 1.0, else 1.3. The 24h pooled
//    quantiles overstate dispersion for quiet markets (most of them) and
//    understate it for live ones. Applied only when a market has >= 480
//    minutes of history: young markets (p00's 395-minute cohort) are in a
//    price-discovery phase where rho is meaningless, and conditioning them
//    HURT p00 by 0.012; the 480-minute guard flips p00 to a gain.
// 2. CROSS-MARKET SUM DRIFT. The 24 Yes prices are probabilities of mutually
//    exclusive outcomes, so their sum should sit near 1 (observed full-
//    universe truth means 1.00-1.05), yet checkpoint sums run 0.81-1.09 and
//    drift back: after the sum crashed to 0.81 at p02's checkpoint it rose
//    0.084 over the window. So the median drifts toward the sum's target:
//    shift = 0.5 * (S - 1) / n_markets * (1 - exp(-h/120)), subtracted from
//    the median row. Small (at most ~0.004 per market) and bounded.
// 3. POST-UP-JUMP MEDIAN DRIFT. After a 1-minute rise >= 0.02 the price
//    reverts hard (pooled over p05 history, n=253: mean -0.138, median
//    -0.101 over the next 240 min; after a fall of the same size, almost
//    nothing). So after such a jump the median alone is shifted down by
//    0.1 * (1 - exp(-h/60)); the tails are NOT shifted (shifting them down
//    was tested and is much worse: the jump may be real news, so the upper
//    tail must stay). The monotonicity clamp usually pins the shifted median
//    to the 25% quantile, which is why the score is flat in A from 0.05 to
//    0.20; A=0.1 is the centre of that flat region.
//
// Sweep on the six practice cases (backtest, same harness as v2): v2 scores
// 0.310602 (p00 0.4818, p01 0.3148, p02 0.4235, p03 0.3436, p04 0.2498,
// p05 0.1117). Vol conditioning alone: 0.306175. + sum drift: 0.304950.
// + median drift: 0.304230 (p00 0.4735, p01 0.3112, p02 0.4089, p03 0.3354,
// p04 0.2490, p05 0.1072) — better than v2 on ALL SIX cases, -0.0064 (-2.1%),
// with each component adding on top of the last. VERIFIED on this file
// itself: running main.ts on the six practice inputs scores exactly
// 0.304230 with the same per-case numbers, so the TypeScript matches the
// Python model; two runs are byte-identical and each case takes ~1 s
// (cap 8 s). Rejected after honest testing: horizon-decay of widths (worse at every c), lower-half and
// median+25% jump drifts at A=0.1 (much worse), 72h/24h window mix and
// post-jump drift on all quantiles (already rejected in v2).
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
/** Shrinkage of each empirical quantile toward zero width, by tau
 *  (0.05 / 0.25 / 0.50 / 0.75 / 0.95): the lower tail is kept wide (falls are
 *  persistent), the upper tail is shrunk hard (rises revert), the middle is
 *  shrunk hardest (most mass sits at zero change). */
const SHRINK = [0.35, 0.8, 0.85, 0.9, 0.7];
/** Minimum minutes of history for volatility conditioning to apply. */
const MIN_HIST = 480;
/** 1-minute rise that counts as an up-jump for the median drift. */
const JUMP_THRESH = 0.02;
/** Median-drift size and time constant (minutes) after an up-jump. */
const JUMP_A = 0.1;
const JUMP_H = 60;
/** Sum-drift strength and time constant (minutes); target sum is 1. */
const SUM_K = 0.5;
const SUM_H = 120;

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
    out.push(QUANTILES.map((t, j) => quantileSorted(ch, t) * (1 - (SHRINK[j] as number))));
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

/**
 * Volatility-conditioning width multiplier for one market: rho is the last
 * 60 minutes' realized absolute move over the mean hourly realized absolute
 * move of the last 24 hours. Quiet markets get narrower bands, live markets
 * wider ones; markets with too little history are left unconditioned (1.0).
 */
function volMult(prices: number[]): number {
  const n1 = prices.length - 1;
  if (n1 < MIN_HIST) return 1;
  let r60 = 0;
  for (let i = Math.max(0, n1 - 60); i < n1; i++) r60 += Math.abs((prices[i + 1] as number) - (prices[i] as number));
  const w24 = Math.min(HIST_WIN, n1);
  let r24 = 0;
  for (let i = n1 - w24; i < n1; i++) r24 += Math.abs((prices[i + 1] as number) - (prices[i] as number));
  const r24h = r24 / (w24 / 60);
  const rho = r60 / Math.max(r24h, 1e-9);
  return rho < 0.5 ? 0.6 : rho < 1 ? 1.0 : 1.3;
}

/** The forecast of one market: horizon rows of five quantiles. */
function forecast(m: CaseMarket, horizon: number, sumPerMarket: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const p = m.prices;
  const x = p[p.length - 1] as number;
  const eq = empQuantiles(p);
  const g = volMult(p);
  const last1 = p.length >= 2 ? (p[p.length - 1] as number) - (p[p.length - 2] as number) : 0;
  const jumped = last1 >= JUMP_THRESH;
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const q = interpQ(eq, h).map((v) => v * g);
    if (jumped) q[2] = (q[2] as number) - JUMP_A * (1 - Math.exp(-h / JUMP_H));
    q[2] = (q[2] as number) - sumPerMarket * (1 - Math.exp(-h / SUM_H));
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
  const sumPerMarket = (SUM_K * (c.markets.reduce((s, m) => s + (m.prices[m.prices.length - 1] as number), 0) - 1)) / c.markets.length;
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes, sumPerMarket) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
