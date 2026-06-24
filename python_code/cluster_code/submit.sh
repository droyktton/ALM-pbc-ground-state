#!/bin/bash
#SBATCH --job-name=alm
#SBATCH --array=0-99           # 100 tasks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=logs/alm_%A_%a.out

# Directorio de trabajo (equivalente a -cwd)
#SBATCH --chdir=.

# Exportar variables de entorno
#SBATCH --export=ALL

#Cola/partición
#SBATCH --partition=knl_tacc


SEEDS_PER_TASK=100              # 100 tasks × 100 seeds = 10000 total samples
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


