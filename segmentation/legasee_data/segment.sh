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
cp ~/data/extract.py ./

echo "Directory Prepared - Contents:"
ls -l
echo "Running segment.py"
python segment.py
cp segments.csv ~/data/
echo "!!! FINISHED !!!"
echo "lets snip the audio up, chop chop chop"
python extract.py
echo "zipping then copying files back to home"
zip -r ./segments.zip segments/
mv segments.zip ~/data/

ls ~/data/ -l
echo "WE ARE DONE, BYE"
