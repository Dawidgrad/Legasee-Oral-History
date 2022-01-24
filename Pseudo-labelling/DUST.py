'''
Implementation of the process described in the paper: Unsupervised Domain Adaptation For Speech Recognition
via Uncertainty Driven Self-Training: https://www.merl.com/publications/docs/TR2021-039.pdf
'''
import argparse
from typing import Dict
import torch
from Levenshtein import distance as levenshtein_distance
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from os.path import join
from os import listdir as ls
import numpy as np
import pandas as pd
import audio_proc # local dataset class and audio functions
from tqdm import tqdm

def lev_distance(a, b):
    return levenshtein_distance(a, b)

def enable_dropout(model:Wav2Vec2ForCTC):
    '''sets dropout layers to train'''
    num = 0
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            num += 1
    print(f'{num} dropout layers enabled')


def forward_pass(model:Wav2Vec2ForCTC, batch:Dict[str, torch.Tensor]):
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in batch.items()}).logits
    return logits

def get_determenistic_predictions(model:Wav2Vec2ForCTC, proc:Wav2Vec2Processor, dataset:audio_proc.dataset, batch_size:int):
    """
    Performs a deterministic forward pass of the model on the dataset.
    """
    labels = []
    for i in tqdm(range(0, len(dataset), batch_size), desc='Deterministic Predictions'):
        batch = dataset[i:i+batch_size]
        logits = forward_pass(model, batch)
        labels.append(audio_proc.greedy_decode(logits, proc))
    # flatten the list
    labels = [item for sublist in labels for item in sublist]
    return labels

def stack_batch(batch:Dict[str, torch.Tensor], monte_carlo:int):
    return {k: torch.cat([v for _ in range(monte_carlo)], dim=0) for k, v in batch.items()}

def get_levenstein_batch(labels, gold):
    '''
    Gets the max, avg, and variance of the levenstein distance (normalized by the length of the label) between the predictions and the gold (gold = deterministic forward pass)
    '''
    lev = [lev_distance(gold, label) for label in labels]
    # normalize based on label length
    levs = [lev[i] / len(gold) for i in range(len(labels))]
    max_l = max(levs)
    avg_l = sum(levs) / len(labels)
    var_l = sum([(l - avg_l)**2 for l in levs]) / (len(labels) - 1) 
    return max_l, avg_l, var_l

def get_stochastic_predictions(model:Wav2Vec2ForCTC, proc:Wav2Vec2Processor, dataset:audio_proc.dataset, monte_carlo:int, csv:pd.DataFrame):
    '''
    Performs K forward passes for each datapoint w/ dropout enabled and gets the levenstein distance between predictions w/ and w/out dropout
    ensure model has dropout enabled
    '''
    max_l = []
    avg_l = []
    var_l = []

    for i in tqdm(range(0, len(dataset)), desc='Stochastic Predictions'):
        batch = dataset[i:i+1]
        #print(batch['input_values'].shape)
        sbatch = stack_batch(batch, monte_carlo)
        #print(sbatch['input_values'].shape)
        logits = forward_pass(model, sbatch)
        labels = audio_proc.greedy_decode(logits, proc)
        gold = csv.iloc[i]['predictions'] 
        m, a, v = get_levenstein_batch(labels, gold)
        max_l.append(m)
        avg_l.append(a)
        var_l.append(v)

    return max_l, avg_l, var_l
    

def main(args):
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wav2Vec2ForCTC.from_pretrained(args.w2v)
    proc = Wav2Vec2Processor.from_pretrained(args.proc)
    model.to(device)
    # load csv
    if args.skip_preds == False:
        csv = pd.read_csv(args.csv)
        dataset = audio_proc.dataset(csv, args.audio_dir, args.audio_i, proc)
        # get deterministic predictions
        model.eval()
        labels = get_determenistic_predictions(model, proc, dataset, args.batch_size)
        # save the labels
        csv['predictions'] = labels
        csv.to_csv(args.csv_out, index=False)
    else:
        print('-- Skipping deterministic predictions --')
        csv = pd.read_csv(args.csv_out)
        dataset = audio_proc.dataset(csv, args.audio_dir, args.audio_i, proc)

    enable_dropout(model)
    # get stochastic predictions
    max_l, avg_l, var_l = get_stochastic_predictions(model, proc, dataset, args.monte_carlo, csv)
    # save the results
    csv['max_l'] = max_l
    csv['avg_l'] = avg_l
    csv['var_l'] = var_l
    csv.to_csv(args.csv_out, index=False)
    print(f'Saved results to {args.csv_out}')
    print(f'WE ALL DONE BBY')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v', help='path to pretrained model', default='../../Legasee-Oral-History/ASR_systems/wav2vec2/train/wav2vec2_7')
    parser.add_argument('--proc', help='path to pretrained processer for w2v', default='../../Legasee-Oral-History/ASR_systems/wav2vec2/train/wav2vec2-large-robust-ft-libri-960h_proc')
    parser.add_argument('--csv', help='path to csv file with data', default='../freds-lowconfidence.csv')
    parser.add_argument('--csv_out', help='path to csv file with predictions', default='fred_pred.csv')
    parser.add_argument('--audio_dir', help='path to audio directory', default='../segments')
    parser.add_argument('--audio_i', help='index of audio file in csv', default=1)
    parser.add_argument('--batch_size', help='batch size', default=25)
    parser.add_argument('--monte_carlo', help='number of monte carlo samples', default=10)
    parser.add_argument('--skip_preds', help='skip deterministic predictions (if they\'ve already been done)', default=False)

    args = parser.parse_args()

    main(args)
