import json
import kenlm
import numpy as np
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from os import listdir
import soundfile as sf
from multiprocessing import Pool, cpu_count
import gzip
import shutil
import urllib.request

##Checks if key is in config and that it does not == None; Returns bool
def isthere(config, string):
    return True if string in config and config[string] != None else False


def save_to_config(config, field, value):
    cur_config = load_config(config['config'])
    cur_config[field] = value
    with open(config['config'], 'w') as jsF:
        json.dump(cur_config, jsF)
    print('--- Config file updated ! ---')

def load_config(path, args=None) -> dict:
    with open(path, 'r') as f:
        config = json.loads(f.read())
    if args == None: 
        return config
    else: ### config values are set as default if they are not prescent in args ###
        for key in config:                  
            if key not in args or args[key] == None:
                args[key] = config[key]
        return args

def get_model(config):
    ls = listdir()
    if  isthere(config, 'model_name') and config['model_name'] in ls and config['model_name']+'_processor' in ls:
        model =  Wav2Vec2ForCTC.from_pretrained(config['model_name'])
        processor = Wav2Vec2Processor.from_pretrained(config['model_name']+'_processor')
        return model, processor
    
    elif isthere(config, 'model'): 
        model =  Wav2Vec2ForCTC.from_pretrained(config['model'])
        processor = Wav2Vec2Processor.from_pretrained(config['model'])
        if isthere(config, 'model_name'):
            model.save_pretrained(config['model_name'])
            processor.save_pretrained(config['model_name']+'_processor')
        else:
            print(f'--- Please provide model_name in config or args to save pre_trained model: {model} locally --')
        return model, processor
    else:
        raise Exception('--- Please provide pre-trained "model" for HuggingFace in args or config file ---')


def kenlm_decoder(config, processor):
    if isthere(config, 'ARPA_Path'):
        vocab_dict = processor.tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
        vocab = []
        for _, token in sort_vocab:
            vocab.append(token.lower())
        vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '
        alpha = 0.5 if isthere(config, 'alpha') == False else config['alpha']
        beta = 1.0 if isthere(config, 'beta') == False else config['beta']
        decoder = build_ctcdecoder(vocab, kenlm_model_path=config['ARPA_Path'], alpha=alpha, beta=beta)
        return decoder, len(vocab)
    else:
        print('--- ARPA Path not given (specify in config or args), searching for "4gram_big.arpa" in current directory instead! ---')
        ls = listdir()
        if "4gram_big.arpa" in ls:
            config['ARPA_Path'] = "4gram_big.arpa" 
            print('--- Found ARPA in path! Updating config ! ---')
            save_to_config(config, 'ARPA_Path', config['ARPA_Path'])
            return kenlm_decoder(config, processor)
        else:
            print('--- "4gram_big.arpa" not found in current directory, attempting to fetch file ---')
            urlarpa = 'https://kaldi-asr.org/models/5/4gram_big.arpa.gz'
            print(f'--- Attempting to fetch: --- {urlarpa}')
            fn, _ = urllib.request.urlretrieve(urlarpa, '4gram_big.arga.gz')
            print(f'--- Succefully fetched ARPA file {fn} ---')
            
            with gzip.open(fn, 'rb') as fin:
                with open('4gram_big.arpa', 'wb') as fout:
                    shutil.copyfileobj(fin, fout)

            print('--- ARPA extracted in currect directory, config file updated ---')
            config['ARPA_Path'] = "4gram_big.arpa"
            save_to_config(config, 'ARPA_Path', config['ARPA_Path'])

            return kenlm_decoder(config, processor)

def process_wav(wav, config):
    speech, _ = sf.read(wav)
    speech = speech[:,0]
    print(f"Frequency: {_}")
    if config != None:
        if isthere(config, 'inference_batch'):
            batch = config["inference_batch"]
            print(f'--- Batching speech array into: {batch} split ---')
            speech_batched = np.array_split(speech, len(speech) // batch)
            return speech_batched
        else:
            print(f'Specify "inference_batch" in args or config to segment numpy audio data')
    return speech

###Not Functional due to error ):
def multiprocess_decode_batch(decoder, logits, config):  
    cpus = cpu_count() if isthere(config, 'cpu_count') == False else config['cpu_count']
    print(f'--- Beam search decoding with cpu count: {cpus} ---')
    beam_width = 100 if isthere(config, 'beam_width') == False else config['beam_width']
    print(f'--- Using Beam Width: {beam_width} ---')
    
    with Pool() as pool:
        transcription = decoder.decode_batch(pool, logits, beam_width)

    print('\n--- Finished Decoding ---\n')
    return transcription
