#!/bin/bash
#SBATCH --job-name=alm
#SBATCH --array=0-99           # 100 tasks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/alm_%A_%a.out

SEEDS_PER_TASK=10              # 100 tasks × 10 seeds = 1000 total samples
SEED_START=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))
SEED_END=$(( SEED_START + SEEDS_PER_TASK ))

mkdir -p results logs

python alm_sim_sample.py \
    --seed_start $SEED_START \
    --seed_end   $SEED_END \
    --L          8192 \
    --Delta      1.0 \
    --c          0 \
    --n_min      2 \
    --n_max      50 \
    --outdir     results
