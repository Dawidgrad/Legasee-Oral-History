from tqdm import tqdm
from multiprocessing import Pool
from os import listdir as ls
import pandas as pd 
import numpy as np
import process_textgrid as pt
import argparse

def get_wav_files(path:str) -> list:
    return sorted([el for el in ls(path) if el.endswith('.wav')])

files = get_wav_files(None)

def proc_file(i:int) -> pd.DataFrame:
    utts = pt.Utterances(files[i])
    return utts.save_utterances()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, help='Number of cores to use for multiprocessing', default=4)
    parser.add_argument('--df_name', type=str, help='output name for dataframe', default='SBNC.df')
    args = parser.parse_args()
    cores = args.cores

    with Pool(cores) as p:
        dfs = list(tqdm(p.imap(proc_file, range(len(files))), total=len(files)))

    df = pd.concat(dfs)
    df.to_csv(args.df_name, index=False)
    print(f' --- Saved dataframe to {args.df_name} ---')
    print(' --- Saved audio segments to: out/---')