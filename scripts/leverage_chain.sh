#!/bin/bash
# Sequential GPU chain, in order of leverage:
#   1. Train scaled-up policy net (192ch / 6 blocks, 28M positions)
#   2. Re-extract cheat features + per-move sequences with the new net
#      (backs up the small-net features first)
#   3. Retrain aggregate detector + train transformer detector
#   4. Strength test vs Stockfish Elo ladder
# Each stage logs to logs/. Stages run regardless of earlier failures
# (each is independently useful); check logs per stage.

cd "$(dirname "$0")/.."
P=.venv/bin/python
mkdir -p logs

echo "[chain] stage 1: train large policy net  $(date)"
$P -u src/train_policy.py --channels 192 --blocks 6 --policy-ch 64 \
    --max-positions 28000000 --batch 1024 --lr 2e-4 --epochs 14 \
    --out models/policy_cnn_large.pt > logs/train_large.log 2>&1

echo "[chain] stage 2: re-extract cheat features + sequences  $(date)"
cp data/processed/cheat_features.parquet \
   data/processed/cheat_features_smallnet.parquet 2>/dev/null
$P -u src/extract_cheat_features.py --max-games 30 --sequences \
    > logs/cheat_seq.log 2>&1

echo "[chain] stage 3a: retrain aggregate detector  $(date)"
$P -u src/train_cheat_detector.py > logs/detector_largenet.log 2>&1

echo "[chain] stage 3b: train transformer detector  $(date)"
$P -u src/train_cheat_transformer.py > logs/train_transformer.log 2>&1

echo "[chain] stage 4: strength test vs Stockfish  $(date)"
$P -u src/strength_test.py --elo-list 1320,1600,1900 --games 12 --sims 300 \
    > logs/strength.log 2>&1

echo "[chain] ALL DONE  $(date)"
