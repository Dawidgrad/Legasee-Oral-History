#!/bin/bash
#SBATCH --mem=50G
#SBATCH --time=03:30:00
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu
#SBATCH --gpus-per-node=4




module load Anaconda3/2019.07
source activate Legasee

echo "Okay, throw me some numbers"

python main.py --gpus 4 -batch 4 -beam 200 -conf 8
python main.py -downstream -punct -ner

echo "WE ARE DONE, BYE"

