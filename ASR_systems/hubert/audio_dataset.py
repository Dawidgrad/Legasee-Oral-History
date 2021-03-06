import re
import torch
import torchaudio
from torch.utils.data import Dataset
import soundfile as sf
import pandas as pd
from transformers import Wav2Vec2Processor
import numpy as np

class dataset(Dataset):
    def __init__(
        self, 
        csv:pd.DataFrame, 
        audio_path:str, 
        wav2cecProc:Wav2Vec2Processor,
        audio_i=1,
        label_i=2, 
        f_extension='', 
        header=None
        ):

        self.data_frame = csv
        self.data_path = audio_path if audio_path[-1] == '/' else f'{audio_path}/'
        self.audio_i = audio_i
        self.label_i = label_i
        self.wav2vecProc = wav2cecProc
        self.f_extension = f_extension

    def __len__(self):
        return len(self.data_frame)

    def _get_audio_path(self, index):
        return self.data_path + self.data_frame.iloc[index, self.audio_i].strip() #+ self.f_extension
        
    def _get_label(self, index):
        return self.data_frame.iloc[index, self.label_i]

    def __getitem__(self, index):
        speech_path = self._get_audio_path(index)
        label = self._get_label(index)
        speech, _ = torchaudio.load(speech_path) # sampling rate is assumed to be 16000
        return speech[0].numpy(), label 

    def _batch_to_dict(self, batch):
        return {
            'input_values':batch[0],
            'attention_mask':batch[1],
            'input_ids':batch[2],
        }

    def collocate(self, batch):
        speech = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        speech = self.wav2vecProc(speech ,padding="longest", sampling_rate=16000, return_tensors='pt')
        input_values = speech['input_values']
        attention_mask = speech['attention_mask']
        with self.wav2vecProc.as_target_processor():
            labels = self.wav2vecProc(labels, padding="longest", return_tensors='pt')
        
        input_ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100) # ignore loss for padding
        
        return self._batch_to_dict([input_values, attention_mask, input_ids])
        
        
