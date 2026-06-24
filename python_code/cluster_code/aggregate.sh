#!/bin/bash
#SBATCH --job-name=alm_agg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/aggregate.out

python aggregate.py
