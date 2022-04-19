#!/bin/bash
#SBATCH --mem=50G
#SBATCH --time=03:30:00
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu
#SBATCH --gpus-per-node=4




module load Anaconda3/2019.07
source activate Legasee

echo "Okay, lets go"

python main.py --gpus 4 -batch 8 -beam 50 
python main.py -downstream -punct -ner

echo "Okay, bye"

