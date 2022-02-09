'''
Evaluates the model on the test set and saves the results to a csv file.
'''
from ast import parse
import chunk
from distutils import text_file
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
from os.path import join
import soundfile as sf
import multiprocessing

import kenlm
from pyctcdecode import build_ctcdecoder
from get_segments import process

import multiLM

from os.path import exists

from jiwer import wer, cer

def load_model(args):
    model = Wav2Vec2ForCTC.from_pretrained(args.model)
    processor = Wav2Vec2Processor.from_pretrained(args.processor)
    model.eval()
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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

def kenlm_decoder(arpa, vocab, alpha=0.5, beta=0.8):  
    alpha = alpha
    beta = beta
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    return decoder

def decode_lm(logits, decoder):
    ''' decode logits to text '''
    #decoded = decoder.decode(logits, beam_width=1000).strip().upper()
    with multiprocessing.get_context('fork').Pool() as pool:
        decoded = decoder.decode_batch(pool, logits, beam_width=500)
    decoded = " ".join(decoded).strip().upper()
    return decoded

def predict(args, model, processor, chunks, vocab, decoder=None):
    batch_size = args.batch_size
    #max_ids = []
    logit_list = []

    max_ids = []
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            inp = processor(batch, padding='longest', return_tensors='pt', sampling_rate=16000)
            out = model(**{k: v.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for k, v in inp.items()}).logits.cpu().numpy()
            logging(args.log_pth, f'Batch {i}-{i+batch_size} of {len(chunks)} -- {out.shape}')
            logit_list.append(out.reshape(-1, 32))
            out = np.argmax(out, axis=-1).reshape(-1)
            max_ids.append(out)
    
    logits = np.concatenate(logit_list, axis=0)
    if decoder is not None:
        decoded = decode_lm(logit_list, decoder)
        print(decoded)
        print('-'*20)
    else:
        decoded = " ".join(processor.batch_decode(max_ids))
        # remove double or more spaces
        decoded = " ".join(decoded.split())
  
    #output = "".join(processor.batch_decode(max_ids)).strip()
    logging(args.log_pth, decoded+'\n\n')
    return decoded



def main(args):
    model, proc = load_model(args)
    print(f'--- Model loaded: {args.model} ---')
    csv = pd.read_csv(args.csv)

    vocab = get_vocab(proc)

    if args.kenlm != '':
        if args.kenlm2 == '':
            decoder = kenlm_decoder(args.kenlm, vocab)
            print(f'--- KenLM loaded: {args.kenlm} ---')
        else:
            print(f'--- Loading multiple KenLM models: {args.kenlm} and {args.kenlm2} ---')
            decoder = multiLM.build_ctcdecoder(multiLM.get_vocab(), [args.kenlm, args.kenlm2], alpha=0.38, beta=0.65)
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
        output = predict(args, model, proc, chunks, vocab, decoder)
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
    parser.add_argument('--model', type=str, help='Path to model file', default='../train/allfinallegasee')
    parser.add_argument('--processor', type=str, help='Path to w2vec processor file', default='../train/wav2vec2-large-robust-ft-libri-960h_proc')
    parser.add_argument('--csv', type=str, help='Path to csv file denoting wav files and corresponding text files', default='data.csv')
    parser.add_argument('--audio_dir', type=str, help='Path to audio directory', default='./audio/')
    parser.add_argument('--text_dir', type=str, help='Path to text directory', default='./plain_text/')
    parser.add_argument('--log_pth', type=str, help='Path to logs directory', default='lm.log')
    parser.add_argument('--output_dir', type=str, help='Path to output directory (NONE if saving of outputs is not needed)', default='')
    parser.add_argument('--max_seg_len', type=int, help='Maximum length of audio segments', default=600000)
    parser.add_argument('--batch_size', type=int, help='Batch size for inference', default=1)
    parser.add_argument('--kenlm', type=str, help='Path to kenlm  arpa file', default='4gram_big.arpa')
    parser.add_argument('--kenlm2', type=str, help='Path to kenlm  arpa file', default='')
    parser.add_argument('--save_csv', type=str, help='Path to csv file to save results', default='results.csv')

    main(parser.parse_args())

