#!/bin/bash
#SBATCH -C knl
#SBATCH -N 4
#SBATCH -q debug
#SBATCH -t 2
#SBATCH -J train-DummyAE
#SBATCH -o logs/%x-%j.out
#SBATCH -D $WORKDIR

# Get node IDs
touch /dev/null > $WORKDIR/mynodes.txt
scontrol show hostname $SLURM_JOB_NODELIST > $WORKDIR/mynodes.txt

# Setup software
module load pytorch

# Run the training
srun -l python test_DummyAE_rpc.py
