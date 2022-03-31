from ast import List
import re
from typing import Dict
import torch
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
import pandas as pd
from os.path import join
from transformers import Wav2Vec2Processor
import multiprocessing

class dataset():
    def __init__(self, csv:pd.DataFrame, audio_path:str, audio_i:int, proc:Wav2Vec2Processor):
        self.csv = csv
        self.audio_path = audio_path
        self.audio_i = audio_i
        self.wav2vecProc = proc

    def __len__(self):
        return len(self.csv)

    def __load_item(self, audio_f):
        speech, _ = torchaudio.load(join(self.audio_path, audio_f))
        return speech[0].numpy()

    def __collocate(self, batch) -> Dict[str, torch.Tensor]:
        speech = self.wav2vecProc(batch, padding="longest", sampling_rate=16000, return_tensors='pt')
        return {
            'input_values': speech['input_values'],
            'attention_mask': speech['attention_mask']
        }

    def __getitem__(self, index):
        items = []
        for item in self.csv.iloc[index, self.audio_i]:
            items.append(self.__load_item(item))
        return self.__collocate(items)
        

def greedy_decode(logits:torch.Tensor, proc:Wav2Vec2Processor):
    """
    Decodes the logits using greedy decoding.
    """
    pred_ids = logits.argmax(-1)
    return proc.batch_decode(pred_ids)

def lm_decode(logits:torch.Tensor, decoder):
    """
    Decodes the logits using language model decoding.
    returns list of strings
    """
    with multiprocessing.get_context('fork').Pool() as pool:
        decoded = decoder.decode_batch(pool, logits.numpy(), beam_width=100)
    return [x.strip().upper() for x in decoded]