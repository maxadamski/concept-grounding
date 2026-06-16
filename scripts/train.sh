#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=hgx
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --array=1-20%4

PYTHON=$HOME/var/jax/bin/python

# baseline
#AGENT="final_baseline"
#NAME="b"

# compression
#AGENT="e32"
#NAME="e32"

# autoencoder
#AGENT="a32"
#NAME="a32"

AGENT="kb32"
NAME="kb32"

SEED=$SLURM_ARRAY_TASK_ID
MODEL="${NAME}_s${SEED}"
$PYTHON train.py --ground=0.5 --seed=$SEED --agent=agents/$AGENT.py --model=models/$MODEL.sav >> logs/$MODEL.out
