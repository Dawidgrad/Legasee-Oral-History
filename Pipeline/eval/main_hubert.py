'''
Evaluates the model on the test set and saves the results to a csv file.
'''
from ast import arg, parse
import chunk
from distutils import text_file
from tabnanny import verbose
from typing import List
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
from os.path import join
import soundfile as sf
import multiprocessing
from parallelformers import parallelize
import kenlm
from pyctcdecode import build_ctcdecoder
from get_segments import process
from Levenshtein import distance as levenshtein_distance
import os

from wer_prediction import confidence_prediction

import multiLM

from os.path import exists

from jiwer import wer, cer

def load_model(args):
    model = HubertForCTC.from_pretrained(args.model)
    processor = Wav2Vec2Processor.from_pretrained(args.processor)
    model.eval()
    if args.gpus > 1:
        parallelize(model, num_gpus=args.gpus, fp16=False, verbose='detail')
    else:
        model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    return model, processor

def logging(path, data):
    with open(path, 'a') as f:
        f.write(data + '\n')


def load_txt(txt_path:str) -> list:
    with open(txt_path, 'r') as f:
        lines = f.read()
    return lines.upper().strip() 

def get_vocab(processor):
    vocab_dict = processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
    vocab = []
    for _, key in sort_vocab:
        vocab.append(key.lower())
    vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '
    return vocab

def kenlm_decoder(arpa, vocab, alpha=0.4, beta=0.8):  
    alpha = alpha
    beta = beta
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    return decoder

def decode_lm(logits, decoder, hot_words=None):
    ''' decode logits to text 
        hot_words = List[str]
    '''
    #decoded = decoder.decode(logits, beam_width=1000).strip().upper()
    with multiprocessing.get_context('fork').Pool() as pool:
        decoded = decoder.decode_batch(pool, logits, beam_width=200, hotwords=hot_words)
    decoded = " ".join(decoded).strip().upper()
    return decoded

'''
def decode_lm(logits, decoder):
    #decoded = decoder.decode(logits, beam_width=1000).strip().upper()
    with multiprocessing.get_context('fork').Pool() as pool:
        decoded = decoder.decode_batch(pool, logits, beam_width=250)
    decoded = " ".join(decoded).strip().upper()
    return decoded
'''


def enable_dropout(model:Wav2Vec2ForCTC):
    '''sets dropout layers to train'''
    num = 0
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            num += 1
    print(f'{num} dropout layers enabled')


def get_levenstein_batch(labels, gold):
    '''
    Gets the max of the levenstein distance (normalized by the length of the label) between the predictions and the gold (gold = deterministic forward pass)
    '''
    lev = [levenshtein_distance(gold, label) for label in labels]
    # normalize based on label length
    levs = [lev[i] / len(gold) for i in range(len(labels))]
    max_l = max(levs)
    avg_l = sum(levs) / len(labels)
    return max_l, avg_l

def get_batch_seconds(batch:torch.Tensor) -> float:
    return batch.squeeze().shape[0] / 16000

def get_confidence(args, proc:Wav2Vec2Processor, reference:torch.Tensor, hypothesis:torch.Tensor, wer_predictor:confidence_prediction, batch:torch.Tensor):
    '''
    Calculates the confidence of the model. ONLY WORKING WITH BATCH SIZE OF 1!!!
    '''
    all_max_ids = torch.cat([reference.unsqueeze(0), hypothesis])
    text_outputs = proc.batch_decode(all_max_ids)
    ref, hyp = text_outputs[:1][0], text_outputs[1:]
    max_l, avg_l = get_levenstein_batch(hyp, ref)
    seconds = get_batch_seconds(batch)
    pred_wer = wer_predictor.predict(max_l, avg_l, ref, seconds)
    return pred_wer


def get_hypotheses(args, model, inputs, k=10) -> List[torch.Tensor]:
    '''
    Returns outputted logits from k forward passes with dropout enabled for confidence estimation
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    enable_dropout(model) # enable dropout layers
    with torch.no_grad():
        hypotheses = [model(**inputs).logits.cpu() if args.gpus > 1 else model(**{k: v.to(device) for k, v in inputs.items()}).logits.cpu() for _ in range(k)] # should probs batch this but whatever
    hypotheses = torch.cat(hypotheses, dim=0).argmax(dim=-1)
    model.eval() # disable dropout layers
    return hypotheses


def predict(args, model, processor, chunks, vocab, decoder=None, wer_predictor=None):
    batch_size = args.batch_size
    #max_ids = []
    logit_list = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    max_ids = []
    conf_list = []
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            inp = processor(batch, padding='longest', return_tensors='pt', sampling_rate=16000)
            out = model(**inp).logits.cpu() if args.gpus > 1 else model(**{k: v.to(device) for k, v in inp.items()}).logits.cpu()
         
            if args.confidence == True:
                hypotheses = get_hypotheses(args, model, inp, k=10)
                reference = out.reshape(-1, 32).argmax(dim=-1)
                conf = get_confidence(args, processor, reference, hypotheses, wer_predictor, batch[0]) # only works with batch size of 1 atm
                conf_list.append(conf)
            logging(args.log_pth, f'Batch {i}-{i+batch_size} of {len(chunks)} -- {out.shape}')
            logit_list.append(out.reshape(-1, 32).numpy())
            out = np.argmax(out.numpy(), axis=-1).reshape(-1)
            max_ids.append(out)
    
    #logits = np.concatenate(logit_list, axis=0)
    if decoder is not None:
        decoded = decode_lm(logit_list, decoder, decoder.gazetteer)
        print(decoded)
        print('-'*20)
    else:
        decoded = " ".join(processor.batch_decode(max_ids))
        # remove double or more spaces
        decoded = " ".join(decoded.split())
  
    #output = "".join(processor.batch_decode(max_ids)).strip()
    logging(args.log_pth, decoded+'\n\n')
    avg_conf = sum(conf_list) / len(conf_list) if args.confidence == True else None
    logging(args.log_pth, f'Average confidence: {avg_conf}')
    return decoded

def load_gazetteer(gaz_path:str) -> List[str]:
    if gaz_path is None or not os.path.exists(gaz_path) or not os.path.isfile(gaz_path) or gaz_path == '':
        return None
    else:
        with open(gaz_path, 'r') as f:
            gaz = f.read().split('\n')
        return gaz


def main(args):
    model, proc = load_model(args)
    print(f'--- Model loaded: {args.model} ---')
    csv = pd.read_csv(args.csv)

    vocab = get_vocab(proc)
    gaz = load_gazetteer(args.gazetteer)

    wer_predictor = confidence_prediction() if args.confidence == True else None

    if args.kenlm != '':
        if args.kenlm2 == '':
            decoder = kenlm_decoder(args.kenlm, vocab)
            decoder.gazetteer = gaz # save to namespace for later use
            print(f'--- KenLM loaded: {args.kenlm} ---')
        else:
            print(f'--- Loading multiple KenLM models: {args.kenlm} and {args.kenlm2} ---')
            decoder = multiLM.build_ctcdecoder(multiLM.get_vocab(), [args.kenlm, args.kenlm2], alpha=0.45, beta=0.8)
            decoder.gazetteer = gaz
            print(f'--- LMs loaded: {args.kenlm} and {args.kenlm2} ---')
    else:
        decoder = None

    rslts = []
    outputs = []

    for i in tqdm(range(len(csv))): 
        audio_path = join(args.audio_dir, csv.iloc[i]['Wav_File'])
        txt_path = join(args.text_dir, csv.iloc[i]['Text_File'])
        logging(args.log_pth, f'--- Processing file: {audio_path} ---')
        chunks = process(audio_path, 0, args.max_seg_len)
        output = predict(args, model, proc, chunks, vocab, decoder,wer_predictor)
        text_file = load_txt(txt_path)
        word_error = wer(text_file, output)
        char_error = cer(text_file, output)
        logging(args.log_pth, f'--- WER: {word_error} - CER: {char_error} --- \n\n\n')
        
        rslts.append({
            'Model': args.model.split('/')[-1],
            'Language_Model': args.kenlm.split('/')[-1] if args.kenlm != '' else 'None',
            'Speaker': csv.iloc[i]['Wav_File'].replace('.wav', ''),
            'WER': word_error,
            'CER': char_error
        })
        outputs.append({
            'Speaker': csv.iloc[i]['Wav_File'].replace('.wav', ''),
            'Output': output
        })
        
    save_csv = pd.DataFrame(rslts) if exists(args.save_csv) == False else pd.read_csv(args.save_csv).append(pd.DataFrame(rslts))
    save_csv.to_csv(args.save_csv, index=False)

    if args.output_dir != '' and args.output_dir != None:
        for output in outputs:
            with open(join(args.output_dir, f'{output["Speaker"]}.txt'), 'w') as f:
                f.write(output['Output'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments for evaluation script')
    parser.add_argument('--model', type=str, help='Path to model file', default='../train/hubert_train/hubert_all')
    parser.add_argument('--processor', type=str, help='Path to w2vec processor file', default='../train/hubert-xlarge-ls960-ft_proc')
    parser.add_argument('--csv', type=str, help='Path to csv file denoting wav files and corresponding text files', default='data.csv')
    parser.add_argument('--audio_dir', type=str, help='Path to audio directory', default='./audio/')
    parser.add_argument('--text_dir', type=str, help='Path to text directory', default='./plain_text/')
    parser.add_argument('--log_pth', type=str, help='Path to logs directory', default='lmh2.log')
    parser.add_argument('--output_dir', type=str, help='Path to output directory (NONE if saving of outputs is not needed)', default='gazet')
    parser.add_argument('--max_seg_len', type=int, help='Maximum length of audio segments', default=600000)
    parser.add_argument('--batch_size', type=int, help='Batch size for inference', default=1)
    parser.add_argument('--kenlm', type=str, help='Path to kenlm  arpa file', default='4gram_big.arpa') # 4gram_big.arpa
    parser.add_argument('--kenlm2', type=str, help='Path to kenlm  arpa file', default='')
    parser.add_argument('--save_csv', type=str, help='Path to csv file to save results', default='resultslm.csv')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use (default: 2)', default=2)
    parser.add_argument('--confidence', help='Whether to output confidence predictions', action='store_true')
    parser.add_argument('--gazetteer', type=str, help='Path to gazetteer file', default='')

    main(parser.parse_args())

