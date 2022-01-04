#!/bin/bash
#$ -l gpu=2
#$ -l rmem=26G
#$ -l h_rt=2:00:00
#$ -P rse
#$ -q rse.q


nvidia-smi

module load apps/python/conda
source activate ML

python main.py --batch_size 3 --cores 1
