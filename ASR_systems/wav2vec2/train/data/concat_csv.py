import pandas as pd
import numpy as np
import argparse
from torch.utils import data

vocab = ['<pad>', '<s>', '</s>', '<unk>', '|', 'E', 'T', 'O', 'A', 'I', 'N', 'H', 'S', 'R', 'L', 'D', 'U', 'Y', 'W', 'M', 'C', 'G', 'F', 'P', 'B', 'K', "'", 'V', 'J', 'X', 'Q', 'Z']

def process_text(text):
    text = text.upper().strip()
    text = "".join(el if el in vocab else ' ' for el in text)
    return text

def main(args):
    datasets = pd.read_csv(args.csv)
    print(f'--- {args.csv} loaded ---')
    csvs = []
    for i, csv in enumerate(datasets['csv_path']):
        cur = pd.read_csv(csv, header=None)
        wav_col = datasets.iloc[i]['wav_column']
        audio_path = datasets.iloc[i]['audio_path']
        extension = datasets.iloc[i]['extension']
        txt_col = datasets.iloc[i]['txt_column']
        name = datasets.iloc[i]['name']
        cur[cur.columns[wav_col]] = cur[cur.columns[wav_col]].apply(lambda x: f'{audio_path.strip()}{x.strip()}{extension.strip()}')
        new_csv = {
            'name':name,
            'audio':cur[cur.columns[wav_col]],
            'txt':cur[cur.columns[txt_col]].apply(lambda x: process_text(x))
        }
        csvs.append(pd.DataFrame(new_csv))
        print(f'--- {name} added ---')
    print('--- csvs processed ---')
    all_datasets = pd.concat(csvs)
    print('--- csvs concatenated ---')
    all_datasets.to_csv(args.output, index=False)
    print('--- csv saved ---')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("concatenates multiple csv files and audio folders into a single csv")
    parser.add_argument('--csv', help='path to entire csv file that links to other csv files', default='all_datasets.csv')
    parser.add_argument('--output', help='output name and path for csv output', default='all_data.csv')
    args = parser.parse_args()
    main(args)