#!/bin/bash
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=1
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu

#nvidia-smi

module load Anaconda3/2019.07
source activate ML

python main.py  

echo "WE ARE DONE, BYE"



#// --gpus-per-node=1SBATCH --account=dcs-res SBATCH --partition=dcs-gpu
###SBATCH --gpus-per-node=1
###SBATCH --account=dcs-res
###SBATCH --partition=dcs-gpu