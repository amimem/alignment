#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=2  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=80GB       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:30:00     # DD-HH:MM:SS
#SBATCH -o ./scratch/slurm-%j.out

module load python/3.10.lua libffi cudatoolkit/12.3.2
source ~/align/bin/activate

export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache/huggingface

python ~/alignment/llava.py