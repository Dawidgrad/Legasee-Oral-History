#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=190G
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gpus-per-node=4
#SBATCH --time=10:40:00

nvidia-smi

module load Anaconda3/2019.07
#module load fosscuda/2019b  # includes GCC 8.3 CUDA 10.1
module load CUDA/10.2.89-GCC-8.3.0 # includes GCC 8.3 CUDA 10.2
#module load GCC/8.3.0 gnu compiler gcc
#module load CUDA/10.0.130 just CUDA 10.0
#module load cuDNN/7.6.4.38-gcccuda-2019b
source activate ML

export CXX=g++


torchrun --nnodes=1 --nproc_per_node=4 --standalone main.py --data_path ../data/ --audio_path ../data/ --data_csv_path ../data/ --data_csv_path ../data/Legasee.csv --ckpt ./checksaves/external_end_3/checkpoint.pt

#python main.py

#deepspeed --num_gpus 4 --num_nodes 1 --master_addr machine1 main.py  
#python main.py --gpus 4

#CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO python main.py --gpus 4