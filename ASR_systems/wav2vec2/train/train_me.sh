#!/bin/bash
#$ -l gpu=2
#$ -P rse
#$ -q rse.q
#$ -l rmem=30G
#$ -l h_rt=1:00:00

nvidia-smi

module load apps/python/conda
source activate ML

python main.py --batch_size 3


#########Batch script for running on HPC
