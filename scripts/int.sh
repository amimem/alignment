#!/bin/zsh
source /etc/profile
module load python/3.10.lua libffi cudatoolkit/12.3.2
source ~/.venv/align/bin/activate

export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache/huggingface