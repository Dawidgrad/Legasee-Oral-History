#!/bin/bash
#SBATCH --mem=50G
#SBATCH --time=02:30:00





module load Anaconda3/2019.07
source activate ML

echo "Okay, throw me some numbers"


python main.py -downstream -punct -ner

echo "WE ARE DONE, BYE"

