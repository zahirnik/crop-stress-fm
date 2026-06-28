#!/bin/bash
# SLURM submission placeholder for m03_embeddings.
#
# The original job script targeted a specific HPC environment. To run on
# your cluster, fill in the SBATCH directives below for your scheduler's
# partition, account, time, and (for GPU steps) GPU resource, point the
# venv activation and `cd` at your local checkout, then submit with sbatch.

#SBATCH --job-name=m03_embeddings
#SBATCH --output=logs/m03_embeddings_%j.out
#SBATCH --error=logs/m03_embeddings_%j.err
#SBATCH --partition=<your-partition>
#SBATCH --time=<HH:MM:SS>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<N>
#SBATCH --mem=<NN>G
# Uncomment for GPU steps:
# #SBATCH --gres=gpu:1

set -eo pipefail

# source <path-to-your-venv>/bin/activate
# cd <path-to-your-checkout>

python -u src/m03_embeddings.py "$@"
