import argparse
from genericpath import isdir
from posix import listdir
import models
import model_utils
from os import listdir as ls
from os.path import isdir
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def find_file(path=''):
    cur = sorted(listdir(path if path != '' else None))
    print(f'--- Displaying Options from Directory: {path}')
    print(f'--- [0] Go Back ---')
    for i, el in enumerate(cur):
        print(f'--- [{i+1}] {el} ---')
    selection = input("\nSelect option from list: ")
    selection = selection.strip()
    if selection.isnumeric() == True:
        selection = int(selection)
    else:
        print(" Please enter a number restarting... ")
        find_file(path)

    oldpath = path
    path = path+".." if selection == 0 else path+cur[selection-1]
    if isdir(path):
        return find_file(path=path+'/')
    elif path[-3:] == "wav":
        print(f'using file: {path}')
        return path
    else:
        print('Please use a "wav" file, restarting...')
        return find_file(oldpath)



def main(args):
    config = model_utils.load_config(args['config'], args=args)
    wav2vec2, processor = model_utils.get_model(config)
    model = models.ASR_CTC(wav2vec2, config) 
    model.to(model.device_)
    model.eval()

    decoder, vocab_len = model_utils.kenlm_decoder(config, processor)
    if 'wav' in config and config["wav"] != None:
        path = config['wav']
    else:
        print("\n--- Wav file path not in config or args, please select wav from directory ---")
        path = find_file()
    speech_batches = model_utils.process_wav(path, config)

    logit_list = []
    print(f'{len(speech_batches)} total batches --- running network ---')
    with torch.no_grad():
        for speech in tqdm(speech_batches):
            print(speech.shape)
            input_ = processor(speech, return_tensors='pt', sampling_rate=16000)
            logits = model(input_.to(model.device_)).logits.cpu().numpy()[0]
            logit_list.append(logits)
    logit_list = np.array(logit_list).reshape(-1, vocab_len)

    print('--- Performing Beam Search, this may take some time..... ---')
    #transcription = model_utils.multiprocess_decode_batch(decoder, logit_list, config)
    transcription = decoder.decode(logit_list)
    print()
    print(transcription)
    print()
    yn = input('Would you like to save this transcript to a file? Y/N: ')
    yn = yn.lower().strip()
    if yn == 'y':
        name = input('Enter outut name: ')
        with open(name, 'w') as f:
            f.write(transcription)
        print('--- Written to file! ---')

    yn = input('\nContinue? (Y) or Exit (N): ')
    yn = yn.lower().strip()
    if yn == 'y':
        config['wav'] = None
        main(config)

        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file", default="config_transcribe.json")
    parser.add_argument("--wav", type=str, help="Path to wav file", default=None)
    parser.add_argument("--model", type=str, help="Hugging Face model card", default=None)
    parser.add_argument("--ARPA_Path", type=str, help="Path to ARPA file", default=None)
    parser.add_argument("--inference_batch", type=int, help="size of batched to split wav file into", default=None)
    
    args = parser.parse_args()
 
    main(vars(args))