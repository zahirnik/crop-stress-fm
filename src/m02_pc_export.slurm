#!/bin/bash
# SLURM submission placeholder for m02_pc_export.
#
# The original job script targeted a specific HPC environment. To run on
# your cluster, fill in the SBATCH directives below for your scheduler's
# partition, account, time, and (for GPU steps) GPU resource, point the
# venv activation and `cd` at your local checkout, then submit with sbatch.

#SBATCH --job-name=m02_pc_export
#SBATCH --output=logs/m02_pc_export_%j.out
#SBATCH --error=logs/m02_pc_export_%j.err
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

python -u src/m02_pc_export.py "$@"
