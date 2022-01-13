#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=46G
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00

nvidia-smi

module load Anaconda3/2019.07
source activate ML

python main.py --batch_size 4 --gpus 4 

#2300066
