#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=20G
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
cp ~/data/segments.csv ./
cp ~/data/extract.py ./

echo "Directory Prepared - Contents:"
ls -l
echo "extracting segments"
python extract.py
echo "!!! FINISHED !!!"
echo "zipping then copying files back to home"
zip -r ./segments.zip segments/
mv segments.zip ~/data/
mv segments.csv ~/data/
ls ~/data/ -l
echo "WE ARE DONE, BYE"
