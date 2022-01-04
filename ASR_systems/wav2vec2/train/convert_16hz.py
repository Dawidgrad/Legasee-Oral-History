import subprocess
from os import listdir
from os import system as sys
import argparse
from pandas.io import parsers
from os.path import isdir, isfile	

def main(args):
	dirs = listdir(args.directory)
	if args.output_directory != '' and isdir(args.output_directory) == False:
		print(f'--- Creating Output Directory: {args.output_directory} ---')
		sys(f'mkdir {args.output_directory}')

	for item in dirs:
		pth = item if args.directory == None else args.directory + '/' + item
		if isfile(pth) and item[-(len(args.format.strip())):] == args.format.strip():
			print(item)
			out = args.output_directory + f'/{item}' if args.output_directory != '' else item
			cmd = f'ffmpeg -i {pth} -ab 160k -ac 2 -ar 16000 -vn {out}'
			subprocess.call(cmd, shell=True)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert all audio to 16khz wav files')
	parser.add_argument('--directory', '-d', help='target directory', default='data/openslr/welsh_english_male')
	parser.add_argument('--format', help='format of files, i.e .wav', type=str, default='.wav')
	parser.add_argument('--rate', help='rate of files, i.e 16000', type=int, default=16000)
	parser.add_argument('--output_directory', '-o', help='output directory', type=str, default='data/openslr/welsh_english_male_16')
	args = parser.parse_args()
	main(args)
