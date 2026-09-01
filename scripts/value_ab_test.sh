#!/bin/bash
# A/B: does the scaled value net (4.2M params) beat the small one (0.98M)?
# Identical conditions: same opening book, same game count, same sims, same
# policy net. Only the value checkpoint differs.
cd "$(dirname "$0")/.."
P=.venv/bin/python
ARGS="--elo-list 1900,2200,2500 --games 30 --sims 300 --sf-movetime 0.3"

echo "[ab] run A: large value net  $(date)"
$P -u src/strength_test.py $ARGS \
    --value-model models/position_eval_cnn_large.pt \
    --label large-value > logs/strength_large_value.log 2>&1

echo "[ab] run B: small value net  $(date)"
$P -u src/strength_test.py $ARGS \
    --value-model models/position_eval_cnn.pt \
    --label small-value > logs/strength_small_value.log 2>&1

echo "[ab] DONE  $(date)"
