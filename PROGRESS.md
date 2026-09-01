# Project progress & handoff — 2026-08-24

## Status: leverage chain COMPLETE (ALL DONE 2026-08-24 20:43)

All four stages ran clean on the 2026-08-23 relaunch. Results:

### Stage 1 — large policy net (`models/policy_cnn_large.pt`)
192ch / 6 ResBlocks / 21M params, 28M positions, 14 epochs:
**top-1 54.7% / top-3 83.1% / top-5 91.8%** (small net: 49.6/78.4/88.3).
MCTS auto-prefers this model.

### Stage 2 — cheat features re-extracted with large net
126,853 rows + per-move sequences (52 min uncontended). Small-net
features backed up at `cheat_features_smallnet.parquet`.

### Stage 3 — detectors
| Detector | Player AUC | Notes |
|----------|-----------|-------|
| Aggregate HistGBT, small-net feats (July) | 0.766 | precision 0.37 @ 51% recall |
| **Aggregate HistGBT, large-net feats** | **0.787** | precision 0.56 @ 51% recall, 0.73 @ 25% |
| Transformer on sequences | 0.757 | does NOT beat aggregates (honest negative) |

Bot sanity check still separates: bots 0.53 vs clean humans 0.02.

### Stage 4 — strength vs Stockfish (UCI_LimitStrength, 300 sims/move, 0.05s/move SF)
| SF level | W-D-L | Score | Implied engine Elo |
|----------|-------|-------|--------------------|
| 1320 | 12-0-0 | 1.00 | — |
| 1600 | 12-0-0 | 1.00 | — |
| 1900 | 10-1-1 | 0.88 | ~2240 |
| 2200 | 8-1-3  | 0.71 | ~2350 |
| 2500 | 8-3-1  | 0.79 | ~2730 |

**Estimate: ≈2300–2500 under these conditions.** Caveats: 12 games/level
(noisy); SF limited-strength calibration is unreliable at 0.05s/move
(non-monotonic 2200 vs 2500 result shows it). A rigorous Elo needs longer
time controls. Games: `logs/strength_games.pgn`.

## Full results overview (all committed, repo Gitwijk/Chess, main)

| Component | Result |
|-----------|--------|
| Outcome baseline (HistGBT, Elo+ECO) | 57.6% |
| Value CNN (Stockfish labels, 17 planes) | 85.4% winner acc, val_loss 0.6147 |
| Policy CNN small (128ch/3b, 20M pos) | 49.6% top-1 |
| **Policy CNN large (192ch/6b, 28M pos)** | **54.7% top-1, 83.1% top-3, 91.8% top-5** |
| MCTS engine + play CLI (`python src/play.py`) | **≈2300–2500 Elo vs limited-strength Stockfish** |
| Cheat detector (aggregate, large-net features) | **player AUC 0.787**, precision 0.73 @ 25% recall; bots 0.53 vs clean 0.02 |
| Cheat detector (transformer on sequences) | player AUC 0.757 (aggregates win) |

Labels: 6,000 players via Lichess API → 389 tos_violation, 109 BOTs, 1,627
disabled (excluded). `data/processed/players.parquet`.

## Next-steps plan (review 2026-08-25, nothing currently running)

### Review findings that drive the priorities
| Measurement | Value | Implication |
|---|---|---|
| Value net size | 0.98M params (128ch/3b) | **21× smaller than the policy net (21M)** — and it evaluates every MCTS leaf. Clear weakest link. |
| Policy net | 21M params, 54.7% top-1 | At the supervised-learning ceiling; more data/size has diminishing returns. |
| Batched inference | 0.15 ms/pos vs 1.35 ms at batch-1 | **9× speedup available**; MCTS currently runs batch-1 only. |
| MCTS throughput | 205 sims/s (NN ≈ 54% of time) | Batching → realistically 2–3× more sims/s → stronger play. |
| Eval data | 19.7M positions used = ALL extracted (5% sample of ~200M available) | Re-extract at ~8% → ~28M, matching the RAM cap. |
| Policy data | 42.2M extracted, 28M used | 14M spare, but see ceiling above. |
| Elo measurement | 12 games/level, SF at 0.05 s/move, non-monotonic | Too noisy to verify any improvement. Fix before optimising. |

### Phase A — engine strength (main push, GPU-bound, sequential)
1. **Better Elo yardstick first** (~1 h code). Without it we cannot tell whether
   A2/A3 helped. Longer TC (0.5–1 s/move for SF), 30+ games/level, fixed opening
   book for variety, report ±error bars. Runs unattended.
2. **Scale the value net** (~30 min code + ~20–30 h train). Parameterise
   `train_cnn.py` exactly like `train_policy.py` (channels/blocks/out/cosine LR/
   checkpointing), then train 192ch/6b. Optionally re-extract evals at 8% first
   (~5 h, CPU-only) for 28M positions. Proven recipe: the same scaling gave the
   policy net +5.1 pp.
3. **Batched MCTS** (~2 h code). Collect leaves per iteration with virtual loss,
   one batched forward pass for policy+value. Verify with the A1 harness.

### Phase B — cheap win, CPU-only (can run alongside Phase A)
4. **Opening analysis** (~1–2 h). Win rates per ECO/variation from
   `games.parquet` (24.9M games) split by Elo band and colour; deliver as a
   ranked table + artifact. Long-standing user interest.

### Phase C — optional / research (only if A plateaus)
5. **Joint trunk, two heads** (~1 day). One backbone, value + policy heads,
   alternating batches from the two datasets (they do NOT overlap: eval-DB
   positions have no played move, game positions have no Stockfish score).
   Payoff: one forward pass per node instead of two, plus shared features.
6. **Cheat detector v2** (~half day). Richer per-move features — eval delta vs
   the engine's best move (not just rank), phase tags, opponent-strength
   context — before concluding sequence models don't help (v1: 0.757 vs 0.787).

## Cheat detector v2 (2026-09-01): timing + Elo-normalisation, and they interact

Enabled by the ChessBase corpus on the MyPassport: ~84M Lichess games from
April 2026, Elo 400–3594, **with `[%clk]` clock annotations**. Verified that
the Lichess Elite PGNs carry NO clocks, so this signal was simply unavailable
before. Data: 10,544 labelled players (215 banned, 167 BOTs), Elo-stratified;
167,748 feature rows, 99.8% with usable clock data.

**Ablation on one dataset — this is the clean comparison:**

| Feature set | Player AUC |
|---|---|
| v1 baseline (move quality only) | 0.7483 |
| + timing | 0.7599 |
| + Elo-excess | 0.7536 |
| **+ both (v2)** | **0.7998** |
| play + time only (no Elo metadata) | 0.7136 |

The two groups are **complementary**: together +0.052, versus +0.012 and +0.005
alone. Knowing both "plays above their rating" and "thinking time does not
track position difficulty" is far more discriminative than either signal by
itself. `time_entropy_corr` lands as the #2 feature by permutation importance,
confirming the hypothesis that motivated it.

**Per Elo band** (the point of v2 — v1 only ever saw 2200+):
800–1199 AUC 0.792 · 1200–1599 **0.888** · 1600–1999 0.819 ·
2000–2399 0.672 · 2400–2799 **0.957**.

**Bot control** (167 BOT accounts, never in training): bot-vs-clean AUC
**0.9419**; the average bot outranks 94.2% of clean players. Note the *absolute*
scores are tiny (median 0.0005) because the positive rate here is 1.9% vs 8.6%
in v1 — comparing mean scores across datasets is misleading, so ranking is
reported instead.

**Honest caveats:**
- v2's 0.7998 is **not** comparable to v1's 0.787: different corpora, different
  test sets (39 vs 77 positives). Only the ablation above is apples-to-apples.
- Removing Elo metadata costs 0.086 AUC (0.7998 → 0.7136). A meaningful share
  of the signal is rating *volatility* (`elo_std` is the top feature), which is
  a legitimate but non-behavioural cue. Pure play+time still clears 0.71.
- Precision at 51% recall is only 0.077, far below v1's 0.557 — again a base-rate
  effect (1.9% positives). At 10% recall precision is 1.000.

**Side finding — ban rate is U-shaped:** 3.7% at 400–599, a trough of 1.3%
around 1800, rising again to 3.4% at 2600+. Likely sandbagging at the bottom
and engine use at the top. v1 could not see this at all.

## Value-net A/B (2026-09-01): scaling the value net does NOT help

Pre-registered test: does the 4.2M-param value net beat the 0.98M one in
actual play? Identical conditions — same opening book, same policy net, 30
games at each of SF 1900/2200/2500, only the value checkpoint differs.

| Value net | W-D-L (90 games) | Score | Combined Elo |
|---|---|---|---|
| Large (192ch/6b, 4.2M) | 51-12-27 | 0.633 | **2312** |
| Small (128ch/3b, 0.98M) | 47-21-22 | 0.639 | **2309** |

Difference: **-4.2 Elo, z = -0.08**, 95% CI on the score difference
[-0.146, +0.135]. Indistinguishable — the small net even scored marginally
higher, which is pure noise. Per the rule fixed before running: **scaling the
value net from 0.98M to 4.2M parameters buys no measurable playing strength.**
Third honest negative in this project, after the sequence detector (0.757 vs
0.787) and the opening analysis (+0.32%).

Worth knowing: detecting a genuine 20-Elo difference at this score level needs
~4,700 games, so this harness can only ever resolve large effects.

Why it likely doesn't help: at 300 simulations the tree is shallow and the
policy priors dominate move selection, and the value task itself was already
near its ceiling (val_loss 0.6147 -> 0.6129, a 0.3% gain). **The lever for
strength is more simulations (batched MCTS), not bigger nets.**

Side benefit: both runs agree on ~2310 Elo, tightening the earlier, noisier
"2300-2500" estimate.

## Data inventory (after cleanup 2026-08-29: 29 GB -> 9.1 GB)

**Moved 2026-09-01** to `/Volumes/My Passport Pro/chess-ml/`, with `data` and
`models` as symlinks inside the project — no code path changed. Verified
byte-exact (2,435,474,007 bytes) and by SHA-256 on all six checkpoints.
`.gitignore` needed `/data` and `/models`: the existing patterns match paths
*inside* those directories, not the symlinks themselves. Scripts fail with a
clear `FileNotFoundError` when the drive is not mounted.

| Path | Size | Regenerate with |
|---|---|---|
| `data/processed/policy/` (134 .npz) | 850 MB | `src/extract_policy.py` (~5 h) |
| `data/processed/evals/` (99 shards) | 606 MB | `src/extract_evals.py` (~5 h, needs the .zst below) |
| `data/processed/_parts/` | 352 MB | `src/parse_pgn.py` intermediates |
| `data/processed/games.parquet` | 328 MB | `src/parse_pgn.py` (~1 h) |
| `data/processed/cheat_features*.parquet` | 51 MB | `src/extract_cheat_features.py` (~1 h, GPU) |
| `data/processed/{players,player_counts,opening_stats}.parquet` | 1.5 MB | API fetch / analysis scripts |
| `models/` (6 checkpoints) | 137 MB | **NOT regenerable — days of GPU time** |
| `.venv/` | 960 MB | `pip install` |

**Source PGNs are never copied locally** — read straight from
`/Volumes/Google Drive/Data Science/Chess Data/Lichess/Lichess Elite Database/`.

**Deleted 2026-08-29** (both fully redundant, verified before removal):
- `data/lichess_db_eval.jsonl.zst` (20 GB) — already extracted into
  `evals/` (99 shards / 19,728,345 positions). Re-download if ever needed:
  `curl -O https://database.lichess.org/lichess_db_eval.jsonl.zst`
- `data/processed/positions/` (42 MB) — 12-plane game-outcome dataset,
  superseded by `evals/`; incompatible with every current 17-plane model.

**Known discrepancy (harmless, left in place):** `_parts/` (parsed 2026-06-15)
holds 24,931,473 games vs 24,919,899 in `games.parquet` (merged 2026-06-26) —
59 files short by ~0.03% each. Not duplicates. Most likely the Drive PGNs were
updated between the two dates. `games.parquet` is the newer artifact and is
what every downstream analysis used, so the difference is immaterial; `_parts/`
was kept rather than deleted because the cause is not fully confirmed.

## Gotchas (also in session memory)
- Use `.venv/bin/python` (system python has no numpy/torch); `-u` for logs.
- `data/` fully gitignored (20 GB eval dump inside made `git add -A` hang).
- Save models via `models/_tmp/` + rename (sandbox write quirk).
- PGN scanning: `read_headers` + seek fast-skip; only monthly
  `lichess_elite_YYYY-MM` files (a merged DownloadConflict file duplicates them).
