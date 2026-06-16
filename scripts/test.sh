#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=hgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

PYTHON=$HOME/var/jax/bin/python

for NAME in b; do
for SEED in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  LOG=logs/$NAME.roll.out
  MODEL=models/${NAME}_s${SEED}.sav
  touch $LOG
  grep -q $MODEL $LOG && continue
  $PYTHON test.py --model=$MODEL --seed=42 --episodes=100000 --parallel=4096 --device=cuda >> $LOG
done
done
