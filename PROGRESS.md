# Project progress & handoff — 2026-07-16

## Status: overnight chain RUNNING (soft-paused session)

`scripts/leverage_chain.sh` is running detached as **PID 71522** (started
2026-07-16 11:05 CEST). It survives terminal/session closure. Check on it:

```bash
cat logs/chain.log                 # which stage is active
tail -f logs/train_large.log       # stage 1 live progress
ps -p 71522                        # chain alive?
```

### Chain stages (sequential, single GPU)
| # | Stage | Log | Output | Est. |
|---|-------|-----|--------|------|
| 1 | Train large policy net (192ch/6blocks/21M params, 28M positions, 14 epochs, cosine LR) | `logs/train_large.log` | `models/policy_cnn_large.pt` | ~2h10m/epoch → ~30h total; early stop patience 4 |
| 2 | Re-extract cheat features + per-move sequences with the large net (old features backed up to `cheat_features_smallnet.parquet` first) | `logs/cheat_seq.log` | `data/processed/cheat_features.parquet` + `cheat_features_sequences.parquet` | ~5h |
| 3a | Retrain aggregate detector (HistGBT) | `logs/detector_largenet.log` | `models/cheat_detector.joblib` | minutes |
| 3b | Train transformer detector on sequences | `logs/train_transformer.log` | `models/cheat_transformer.pt` | ~30 min |
| 4 | Strength test vs Stockfish Elo ladder 1320/1600/1900, 12 games each | `logs/strength.log` | Elo estimate + `logs/strength_games.pgn` | ~2h |

Stage 1 progress at pause time: epoch 1/14 done —
`top1=0.4738 top3=0.7611 top5=0.8655` (small net needed 40 epochs for 49.6%).

## Results so far (all committed, repo Gitwijk/Chess, main)

| Component | Result |
|-----------|--------|
| Outcome baseline (HistGBT, Elo+ECO) | 57.6% |
| Value CNN (Stockfish labels, 17 planes) | 85.4% winner acc, val_loss 0.6147 |
| Policy CNN small (128ch/3blocks, 20M pos) | 49.6% top-1, 78.4% top-3, 88.3% top-5 |
| MCTS engine + play CLI | works; self-play produces Najdorf theory (`python src/play.py`) |
| Cheat detector (aggregate, small-net features) | player AUC 0.766, precision 0.73 @ 25% recall; bots 0.55 vs clean 0.02 |

Labels: 6,000 players via Lichess API → 389 tos_violation, 109 BOTs, 1,627
disabled (excluded). `data/processed/players.parquet`.

## When the chain finishes (next session TODO)
1. Read the four stage logs; compare large-net top-1 vs 49.6% small-net.
2. Compare detector AUCs: aggregate small-net (0.766) vs aggregate large-net
   (`logs/detector_largenet.log`) vs transformer (`logs/train_transformer.log`).
3. Read strength test summary (`logs/strength.log`) → engine Elo estimate.
4. Update this file + memory, commit results summary to GitHub.
5. Open ideas after that: value-head joint training with policy (shared
   backbone), MCTS with batched leaf evaluation (much faster search),
   opening win-rate analysis from games.parquet.

## Gotchas (also in session memory)
- Use `.venv/bin/python` (system python has no numpy/torch); `-u` for logs.
- `data/` fully gitignored (20 GB eval dump inside made `git add -A` hang).
- Save models via `models/_tmp/` + rename (sandbox write quirk).
- PGN scanning: `read_headers` + seek fast-skip; only monthly
  `lichess_elite_YYYY-MM` files (a merged DownloadConflict file duplicates them).
