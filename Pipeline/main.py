from ast import Str, parse
from cgitb import text
from posixpath import split
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
from os.path import join
import soundfile as sf
import multiprocessing
import re
import gc 
import datetime

from diarize.speaker_diarization import run_diarization
from model_utils import load_model, apply_softmax, load_decoder, run_model, decode_lm
from diarize.speaker_turns import speaker_turn

import punctuation # not working

from NER_systems import ner

from get_segments import process
import ctc_segmentation
import pickle as pkl
from os.path import exists
import json

def logging(path, data):
    if path != "":
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

def remove_punct(text:List[str]) -> List[str]:
    '''
    punctuation to remove: {! ? , . - : '}
    '''
    return [re.sub(r'[!?,.:-]', '', el) for el in text]

    

def pair_with_timestamps(args, text_spl, output_time_text):
    timestamped_txt = []
    #print(len(text_spl))
    #print(len(output_time_text))
    cur_pos = 0
    for entry in text_spl:
        curtxt = entry if entry.__class__.__name__ == 'str' else " ".join(item['text'] for item in entry).strip()
        if len(remove_punct([curtxt])[0].strip()) == 0: # if empty string other than punctuation (change to remove punct before)
          continue
        txtlen = len(curtxt.split(' '))
        timestamped_txt.append({
            'text': entry,
            'start': output_time_text[cur_pos]['start'],
            'end': output_time_text[cur_pos+txtlen-1]['end']
        })
       
        cur_pos += txtlen
    return timestamped_txt
          

def get_NER_dict(text:str) -> Tuple[List[Dict], str]:
    '''
    Example text: "My name is <PERSON>Rob<\\PERSON>, I am going to <LOCATION>Paris<\\LOCATION>."
    should return: entities:[{'tag': 'PERSON', 'text': 'Rob'}, {'tag': 'LOCATION', 'text': 'Paris'}] e.t.c.
        '''
    # split based on <*>*<\\*> # remove newline chars
    textspl = re.split(r'<[^>]*>', text.replace('\n', ''))
    textspl = " ".join([el.strip() for el in textspl if el.strip() != '']) # textspl = string with tags removed
    # match all tags inside <*>*<\\*>
    tags = re.findall(r'<([^>]*)>', text)
    tags = [x for x in tags if x.startswith('\\') == False]
    # match all NER inside tags i.e. <PERSON>Rob<\\PERSON> -> Rob
    NERs = re.findall(r'<[^>]*>([^<]*)<[^>]*>', text)
    return [{'text':x, 'tag':y} for x,y in zip(NERs, tags)]


def split_NER(text:str) -> List[str]:
    '''
    Example text: "My name is <PERSON>Rob Flynn<\\PERSON>, I am going to <LOCATION>Paris<\\LOCATION>."
    should return: ["My", "name", "is", NER_DICT{"Rob Flynn"...},...]
    splits on tags, and spaces for words which aren't tagged
    '''
    # messy code really
    # remove newline chars, trailing spaces, and double spaces
    text = text.replace('\n', '').strip().replace('  ', ' ')
    textspl = text.split(' ')
    textlst = []
    inside = False
    for t in textspl:
        cur = remove_punct([t])[0] # otherwise "My name is <PERSON>Rob Flynn<\\PERSON>," will cause bug due to comma
        if cur.startswith('<'):
            if cur.endswith('>'):
                textlst.append(t)
            else:
                inside = True
                textlst.append(t)
        elif cur.endswith('>'):
            inside = False
            textlst[-1] += ' ' + t
        elif inside == True:
            textlst[-1] += ' ' + t
        elif inside == False:
            textlst.append(t)
    ner_lst = []
    for t in textlst:
        cur = remove_punct([t])[0]
        if cur.startswith('<'):
            ner_lst.append(get_NER_dict(t))
        else:
            ner_lst.append(t)
    return ner_lst
       


def predict(args, model, processor, chunks, chunk_idxs, vocab, decoder=None):
    out_lst, conf = run_model(args, model, processor, chunks, args.batch_size)   
    decoded_timestamps = decode_lm(args, out_lst, decoder, chunks, chunk_idxs)
    return decoded_timestamps, conf

def add_punctuation(args, model, text:str) -> str:
    return text if args.punctuation != True else model.punctuate(text.strip())

def get_date():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


def build_output(output:Dict) -> Dict:
    sys_out = {}
    sys_out['file_name'] = output['File']
    sys_out['date'] = get_date()
    sys_out['transcriber'] = {
        'type': 'automatic',
        'name': 'SLT-CDT-TEAM-2',
        'confidence': None # add this 
    }
    sys_out['speaker_turns'] = output['Output']['speaker_turns']
    sys_out['plain_output'] = output['Output']['plain_text']
    sys_out['contents'] = []
    for item in output['Output']['segmented_output']:
        sys_out['contents'].append({
            'speaker': item['Speaker_turns'],
            'text': item['text'] if item['text'].__class__.__name__ == 'str' else " ".join(el['text'] for el in item['text']),
            'confidence':None, # add this
            'time': {
                'start': item['start'],
                'end': item['end']
            },
            'entities': None if item['text'].__class__.__name__ == 'str' else item['text']
        })
    return sys_out

def add_model_config_to_args_namespace(args):
    configf = args.model_config
    config = json.load(open(configf, 'r'))
    args.hubert = True if config['large_model'] == True else False
    args.model = config['model_path']
    args.processor = config['processor_path']
    return args

def save_to_json(output:Dict, output_path:str):
    with open(output_path, 'w') as f:
        json.dump(output, f)

def main(args):
    args = add_model_config_to_args_namespace(args)
    model, proc = load_model(args)
    print(f'--- Model loaded: {args.model} ---')

    csv = pd.read_csv(args.csv) # csv file with audio files
    #martin_s, _ = sf.read(args.interviewer_reference) # reference audio file used for diarization
    vocab = get_vocab(proc) # vocab used for forced alignment
    decoder = load_decoder(args, vocab) # kenlm decoder used for shallow fusion

    out_cache = []
    #outputs = []
    
    for i in tqdm(range(len(csv)), desc='Producing Transcript...'): # ASR
        '''
        First compute outputs from ASR model
        ''' 
        audio_path = join(args.audio_dir, csv.iloc[i]['Wav_File'])
        logging(args.log_pth, f'--- Processing file: {audio_path} ---')
        chunks, chunk_idxs = process(audio_path, args.vad_threshold, args.min_chunk_length)
        timestamps_decoded, conf = predict(args, model, proc, chunks, chunk_idxs, vocab, decoder)
        out_cache.append({
            'file': csv.iloc[i]['Wav_File'],
            'output': timestamps_decoded,
            'audio_path': audio_path
        })
    if args.cache_folder != "" and args.cache_folder != None:
        with open(join(args.cache_folder, args.cache_file), 'wb') as f:
            pkl.dump({'vocab':vocab, 'data':out_cache}, f)
        logging(args.log_pth, f'--- Cached output to: {args.cache_folder} ---')


def downstream(args):
    logging(args.log_pth, f'--- Loading Cache ---')
    with open(join(args.cache_folder, args.cache_file), 'rb') as f:
        out_cache = pkl.load(f)
    logging(args.log_pth, f'--- Cache loaded ---')
    martin_s, _ = sf.read(args.interviewer_reference) # reference audio file used for diarization
    vocab = out_cache['vocab'] # vocab used for forced alignment

    rpunct = None if args.punctuation != True else punctuation.RestorePuncts()

    for i, entry in enumerate(tqdm(out_cache['data'], desc='Downstream Processing...')): 
        audio, _ = sf.read(entry['audio_path'])
        
        output_time_text = entry['output']
        output_text = " ".join(el['text'] for el in output_time_text)
        text_output = add_punctuation(args, rpunct, output_text)
        
        logging(args.log_pth, f'--- Punctiation added ---') if args.punctuation == True else None
        
        if args.ner == True:
            tagged_txt = ner.get_named_entities(text_output, './NER_systems/')
            text_spl = split_NER(tagged_txt)
        else:
            text_spl = text_output.split(' ') 
       

        segmented_output = pair_with_timestamps(args, text_spl, output_time_text)
        speaker_turns = run_diarization(martin_s, file=None, wav=audio)
        diarized_output = speaker_turn({'Output': { 'speaker_turns': speaker_turns, 'segmented_output': segmented_output } })
        segmented_output = diarized_output['Output']['segmented_output']

        to_output = build_output({ 
            'File': entry['file'].split('.')[0],
            'Output': {
                'speaker_turns': speaker_turns,
                'segmented_output': segmented_output,
                'plain_text': entry['output']
            }       
        })
        if args.output_dir != '' and args.output_dir != None:
            pth = join(args.output_dir, f'{to_output["file_name"]}.json')
            save_to_json(to_output, pth)
            logging(args.log_pth, f'--- Transcript saved: {pth} ---')
        else:
            print('PLEASE SPECIFY OUTPUT DIR')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments for evaluation script')
    parser.add_argument('-modconf', '--model_config', help='path to config file for ASR model', default='model_config.json')

    parser.add_argument('-gpus','--gpus', type=int, help='Number of GPUs to use (0 = CPU)', default=1)
    parser.add_argument('-fp16','--fp16', default=False, type=bool, help='Whether to use 16-bit precision, only effective on certain GPUs')
    parser.add_argument('-cache','--cache_folder', help='Where to store cache files', default='./cache')
    parser.add_argument('--cache_file', help='Name of file to store cache in', default='out_cache.pkl')
    parser.add_argument('-downstream','--downstream', action='store_true', help='Whether to run downstream processing, requires processed cache file in cache folder')


    parser.add_argument('-a', '--alpha', type=float, default=0.45, help='Alpha weight for language model fusion')
    parser.add_argument('-b', '--beta', type=float, default=0.8, help='Beta value for language model length penalty')
    parser.add_argument('-beam', '--beam_width', type=int, default=200, help='Beam width for beam search (lower is faster, but less accurate)')

    parser.add_argument('-batch','--batch_size', type=int, help='Batch size for inference', default=1)
    parser.add_argument('-min_sec','--min_chunk_length', type=int, help='Minumun length for chunking sequences (for best performance set to ~25', default=25)
    parser.add_argument('-vad','--vad_threshold', type=float, help='Level of aggressiveness of VAD (0-3)', default=0)
    
    parser.add_argument('-punct','--punctuation', help='Whether to add punctuation', action='store_true') 
    parser.add_argument('-ner','--ner', help='Whether to add named entities to the output', action='store_true')
    parser.add_argument('-diar','--diarization', type=bool, help='Whether to perform diarization', default=True) 
    parser.add_argument('-conf','--confidence', type=int, help='Whether to perform confidence estimation 0 = None, Higher is more accurate but N times slower', default=0)

    parser.add_argument('-csv','--csv', type=str, help='Path to csv file denoting wav files to process', default='data.csv')
    parser.add_argument('-d','--audio_dir', type=str, help='Path to audio directory', default='../eval/audio/')

    parser.add_argument('-log','--log_pth', type=str, help='Path to logs directory (blank for no logging)', default='pipeline.log')
    parser.add_argument('-o','--output_dir', type=str, help='Path to output directory (NONE if saving of outputs is not needed)', default='./output')
    

    parser.add_argument('-lm','--kenlm', type=str, help='Path to kenlm  arpa file', default='../eval/4gram_big.arpa')
    parser.add_argument('-lm2','--kenlm2', type=str, help='Path to kenlm second arpa file (blank if none)', default='')

    parser.add_argument('-ref','--interviewer_reference', type=str, help='Path to audio containing interviewer speech for diarization', default='./diarize/martin_all.wav')
    
    args = parser.parse_args()

    if args.downstream == False:
        main(args)
    else:
        downstream(args)





'''

def segment_outputs(args, output:List, lgts:torch.Tensor, audio:np.array, vocab:List) -> List[Dict]:
    logging(args.log_pth, "Segmenting outputs...")
    char_list = vocab
    lpz = apply_softmax(lgts)

    text = [el.lower() if el.__class__.__name__ == 'str' else " ".join(item['text'].lower() for item in el) for el in output]
    text = text if args.punctuation == False else remove_punct(text) # this probably isn't needed actually

    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / lpz.shape[0] / 16000
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, lpz, ground_truth_mat) 
    timestamps = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text) 
    segments = []
    for txt, timings in zip(output, timestamps):
        start, end, confidence = timings
        segments.append({
            'text': txt,
            'start': start,
            'end': end,
        })
    return segments

'''