'''
File for converting all the videos into 16khz wav files
requires ffmpeg to be installed -- sudo apt install ffmpeg

place file in directory containing the folders for each person i.e ['Harriet_Wright', 'Doug_Joyce', 'Frank_Wilson', 'John_Roach', 'Pat_Massett', 'Mervyn_Salter', 'Baden_Singleton', 'Peter_Dunstan', 'Kevin_Fenton', 'Ted_Rogers']

Then run this file with python

This will only work in linux !!
'''
import subprocess
from os import listdir
from os import system as sys

dirs = listdir()
target = 'WAV_FILES'
sys(f'mkdir {target}')
if target in dirs:
	dirs.remove(target)
dirs.remove('convert.py')


for dir in dirs:
	files = listdir(dir)
	for file in files:
		fname = file[:-4].replace(' ','_') + '.wav'
		sys(f'mkdir {target}/{dir}')
		cmd = f'ffmpeg -i {dir}/"{file}" -ab 160k -ac 2 -ar 16000 -vn {target}/{dir}/{fname}'
		subprocess.call(cmd, shell=True)

