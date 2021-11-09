from functools import partial
import json
import numpy as np
import pytorch_lightning as pl
from os import listdir
import soundfile as sf
from datasets import Features, Sequence, Value, load_dataset, load_from_disk, DatasetDict
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    dataset,
    random_split,
    RandomSampler,
    Dataset
)
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class dataset_utils():
    def __init__(self, config):
        self.config=config
        self.training_columns = ['input_values','input_ids']

    def _load_audio(self, doc:dict):
        audio_path =  f"{self.path.strip()}{doc['file'].strip()}.wav" if self.path[-1] == '/' else f"{self.path.strip()}/{doc['file'].strip()}.wav"
        audio, _ = sf.read(audio_path) 
        audio = audio[:,0]  ################### Dual Channel where both channels are equal, if theyre not (they are atm) then change this to an average over channels
        return {'audio': audio}

    def _process_audio(self, doc:dict, proc:Wav2Vec2Processor):
        input_values = proc(doc['audio'], padding="longest", sampling_rate=16000)['input_values']
        input_ids = proc.tokenizer(doc['label'], padding="longest")['input_ids']
        return {
            'input_values':input_values,
            'input_ids':input_ids
        }


    def load_from_csv(self):
        self.dataset = load_dataset("csv", data_files=self.config.data_csv_path, column_names=["id",'file','label'])
        self.path = self.config.data_path
        return self

    def load_dataset(self):
        self.dataset = load_from_disk(self.config.dataset)

    def prepare_split(self, split): #removes columns not needed for training
        return self.dataset[split].remove_columns(
            [el for el in list(self.dataset[split].features) if el not in self.training_columns]
        )

    def map_to_dataset(self):
        df_features = Features(
            {'id': Value('string'), 'file': Value('string'), 'label': Value('string'), 'audio': Sequence(Value('float32'))}
        )
        self.dataset = self.dataset.map(
            self._load_audio,
            features=df_features
        )
        return self

    def map_to_tokenizer(self, processor:Wav2Vec2Processor):
        self.dataset = self.dataset.map(
            partial(self._process_audio, proc=processor),
            batched=True
        )
        return self

    def split_dataset(self):
        self.dataset = self.dataset['train']
        train_testval = self.dataset.train_test_split(test_size=0.2)
        test_val = train_testval['test'].train_test_split(test_size=0.5)
        self.dataset = DatasetDict({
            "train": train_testval['train'],
            'test': test_val['test'],
            'validation': test_val['train']
        })
        return self.dataset

      

class DataModule(pl.LightningDataModule):
    def __init__(self, config, data_manager) -> None:
        super().__init__()
        self.cores = config.cores
        self.batch_size = config.batch_size
        self.data_m = data_manager
        self.train, self.test, self.val = self.data_m.prepare_split('train'), self.data_m.prepare_split('test'), self.data_m.prepare_split('validation')
        self.train.set_format(type='torch', columns=['input_values', 'input_ids'])
        self.test.set_format(type='torch', columns=['input_values', 'input_ids'])
        self.val.set_format(type='torch', columns=['input_values', 'input_ids'])
        
    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        dataset = TensorDataset(self.train['input_values'], self.train['input_ids'])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cores)
    
    def val_dataloader(self):
        dataset = TensorDataset(self.val['input_values'], self.val['input_ids'])
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.cores)
            
    def test_dataloader(self):
        dataset = TensorDataset(self.test['input_values'], self.test['input_ids'])
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.cores)






