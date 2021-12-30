import argparse
import pandas as pd
import numpy as np
import os
import re

TSV = True
FILM_SEP = 'New Film'

def get_films(df):
    films = []
    for i in range(len(df)):
        cur = df.iloc[i]
        if cur['Transcript'][:len(FILM_SEP)] != FILM_SEP: 
            if i == 0:
                films.append([cur['Transcript']])
            else:
                films[-1].append(cur['Transcript'])
        else:
            films.append([])
    return films

def save_films(args, films, fname):
    path = os.path.join(args.output, fname)
    for i, film in enumerate(films):
        text = " ".join(film)
        if len(text.strip()) > 0: 
            text = re.sub(r'\s+', ' ', text) # remove double or more spaces regex
            with open(path + f'_{i+1}.txt', 'w') as f:
                f.write(text)
            print(f'Saved {path + f"_{i+1}.txt"}')

def main(args):
    fname = args.input.split('/')[-1][:-4]
    print(f'{"-"*50}\n{"Parsing: "+fname:^50}\n{"-"*50}')
    df = pd.read_csv(args.input, sep='\t') if TSV == True else pd.read_csv(args.input, sep=',')
    films = get_films(df)
    save_films(args, films, fname)
    print(f'{"-"*50}\n{"Done: "+fname:^50}\n{"-"*50}')

def check_exists(args):
    if not os.path.exists(args.input) or not os.path.exists(args.output):
        raise Exception("Input or output file does not exist")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a TSV file")
    parser.add_argument("-i", "--input", help="Input TSV file", required=True)
    parser.add_argument("-o", "--output", help="Output folder", default='batch_0')
    args = parser.parse_args()
    check_exists(args)

    main(args)
