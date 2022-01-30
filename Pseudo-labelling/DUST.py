'''
Implementation of the process described in the paper: Unsupervised Domain Adaptation For Speech Recognition
via Uncertainty Driven Self-Training: https://www.merl.com/publications/docs/TR2021-039.pdf
'''
import argparse
from cProfile import label
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

from pyctcdecode import build_ctcdecoder
import kenlm

def get_vocab(processor):
    vocab_dict = processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
    vocab = []
    for _, key in sort_vocab:
        vocab.append(key.lower())
    vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '
    return vocab

def kenlm_decoder(arpa, vocab):  
    alpha = 0.125
    beta = 1.0
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    return decoder

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
        logits = model(**{k: v.to(model.device) for k, v in batch.items()}).logits.cpu()
    return logits

def get_determenistic_predictions(model:Wav2Vec2ForCTC, proc:Wav2Vec2Processor, dataset:audio_proc.dataset, batch_size:int, lm_decoder=None):
    """
    Performs a deterministic forward pass of the model on the dataset.
    """
    labels = []
    for i in tqdm(range(0, len(dataset), batch_size), desc='Deterministic Predictions'):
        batch = dataset[i:i+batch_size]
        logits = forward_pass(model, batch)
        if lm_decoder == None:
            labels.append(audio_proc.greedy_decode(logits, proc))
        else:
            labels.append(audio_proc.lm_decode(logits, lm_decoder))
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

def get_stochastic_predictions(model:Wav2Vec2ForCTC, proc:Wav2Vec2Processor, dataset:audio_proc.dataset, monte_carlo:int, csv:pd.DataFrame, lm_decoder=None):
    '''
    Performs K forward passes for each datapoint w/ dropout enabled and gets the levenstein distance between predictions w/ and w/out dropout
    ensure model has dropout enabled
    '''
    max_l = []
    avg_l = []
    var_l = []

    for i in tqdm(range(0, len(dataset)), desc='Stochastic Predictions'):
        batch = dataset[i:i+1]
        gold = csv.iloc[i]['predictions']
        if isinstance(gold, str) == False:
            print(f'\n WE GOT A NAUGHTY LABEL: {gold}') 
            m, a, v = 1000, 1000, 1000 # if it's not a string, i.e nan value, then just set levenshtein to arbitrarily high value
        else:
            sbatch = stack_batch(batch, monte_carlo)
            logits = forward_pass(model, sbatch)
            if lm_decoder == None:
                labels = audio_proc.greedy_decode(logits, proc)
            else:
                labels = audio_proc.lm_decode(logits, lm_decoder)
            labels = [label for label in labels if isinstance(label, str) == True] # remove nan values
            if len(labels) == 0 or len(gold) == 0:
                print('\n oh darn! no labels \n') 
                print(f'\n gold: {gold}')
                print(f'\n labels: {labels}')
                m, a, v = 1000, 1000, 1000 
            else:
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
    model.eval()

    if args.arpa != '' and args.arpa != None:
        lm_decoder = kenlm_decoder(args.arpa, get_vocab(proc))
    else:
        lm_decoder = None

    # load csv
    if args.skip_preds == False:
        csv = pd.read_csv(args.csv)
        dataset = audio_proc.dataset(csv, args.audio_dir, args.audio_i, proc)
        # get deterministic predictions
        labels = get_determenistic_predictions(model, proc, dataset, args.batch_size, lm_decoder)
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
    parser.add_argument('--csv', help='path to csv file with data', default='../fred-s.csv')
    parser.add_argument('--csv_out', help='path to csv file with predictions', default='fred_pred.csv')
    parser.add_argument('--audio_dir', help='path to audio directory', default='../segments')
    parser.add_argument('--audio_i', help='index of audio file in csv', default=1)
    parser.add_argument('--batch_size', help='batch size', default=25)
    parser.add_argument('--monte_carlo', help='number of monte carlo samples', default=10)
    parser.add_argument('--skip_preds', help='skip deterministic predictions (if they\'ve already been done)', default=False)
    parser.add_argument('--arpa', help='path to arpa file', default='4gram_big.arpa')

    args = parser.parse_args()

    main(args)
