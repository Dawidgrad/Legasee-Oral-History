#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00

nvidia-smi

module load Anaconda3/2019.07
source activate ML

echo "okay, throw me some numbers..."
echo "Copying files to scratch"
cd $TMPDIR
rm -rf $TMPDIR/* # make sure dir is empty (it should be :P)
mkdir audio
mkdir segments
cp ~/data/audio/*.wav audio/
unzip ~/data/text_files.zip -d ./
cp ~/data/segment.py ./
cp ~/data/corresponding_audio.csv ./
echo "Directory Prepared - Contents:"
ls -l
echo "Running segment.py"
python segment.py
echo "!!! FINISHED !!!"
echo "zipping then copying files back to home"
zip -r ~/data/segments.zip segments/
mv segments.csv ~/data/
ls ~/data/ -l
echo "WE ARE DONE, BYE"
