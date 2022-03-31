#!/bin/bash
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=1
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu

#nvidia-smi

module load Anaconda3/2019.07
source activate ML

module load CUDA/10.2.89-GCC-8.3.0 # includes GCC 8.3 CUDA 10.2

export CXX=g++


python main_hubert.py --gpus 1 --kenlm "" --confidence

echo "WE ARE DONE, BYE"



#// --gpus-per-node=1SBATCH --account=dcs-res SBATCH --partition=dcs-gpu
###SBATCH --gpus-per-node=1
###SBATCH --account=dcs-res
###SBATCH --partition=dcs-gpu
