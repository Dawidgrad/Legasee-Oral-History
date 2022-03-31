#!/bin/bash
#SBATCH --mem=50G
#SBATCH --time=02:30:00
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu
#SBATCH --gpus-per-node=1




module load Anaconda3/2019.07
source activate ML

echo "Okay, throw me some numbers"

python main.py --gpus 1 -batch 2 -beam 200
python main.py -downstream -punct -ner

echo "WE ARE DONE, BYE"

