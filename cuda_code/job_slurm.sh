#!/bin/bash
#SBATCH --job-name=alm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --output=res_%j.txt

module load cuda/11.1.0-zj6lnzj gcc/8.5.0-5sg556d

seed=$1

./alm $seed
