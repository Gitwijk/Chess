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

## Gotchas (also in session memory)
- Use `.venv/bin/python` (system python has no numpy/torch); `-u` for logs.
- `data/` fully gitignored (20 GB eval dump inside made `git add -A` hang).
- Save models via `models/_tmp/` + rename (sandbox write quirk).
- PGN scanning: `read_headers` + seek fast-skip; only monthly
  `lichess_elite_YYYY-MM` files (a merged DownloadConflict file duplicates them).
