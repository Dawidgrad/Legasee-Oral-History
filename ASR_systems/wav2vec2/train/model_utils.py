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
from transformers import Wav2Vec2Processor
import audio_dataset


def isthere(config, element:str):
    return True if element in config else False


class dataset_utils():
    def __init__(self, config):
        self.config=config

    def load_datasets(self, config):
        if isthere(config, 'data_path'):
            pth = config.data_path if config.data_path[-1] == '/' else f"{config.data_path}/"
            self.train = pd.read_csv(f'{pth}train.csv', header=None)
            self.validate = pd.read_csv(f'{pth}val.csv', header=None)
            self.test = pd.read_csv(f'{pth}test.csv', header=None)
        else:
            print("Please specify 'data_path' in args")
             

    def _save_splits_(self, config):
        if isthere(config, 'data_path'):
            pth = config.data_path if config.data_path[-1] == '/' else f"{config.data_path}/" 
            self.train.to_csv(f'{pth}train.csv', index=None, header=None)
            self.validate.to_csv(f'{pth}val.csv', index=None, header=None)
            self.test.to_csv(f'{pth}test.csv', index=None, header=None)
            print(f'Training splits saved in {pth}')
        else:
            print('Please specify "data_path" in args to save file')

    def create_splits(self):
        if isthere(self.config, 'data_csv_path'):
            train_size = 0.8
            val_size = 0.1
            all_csv = pd.read_csv(self.config.data_csv_path, header=None)
            self.train, self.validate, self.test = np.split(all_csv.sample(frac=1), [int(train_size*len(all_csv)), int((val_size+train_size)*len(all_csv))])
            self._save_splits_(self.config)
        else:
            print('Please specify "data_csv_path" in args')

      

class DataModule(pl.LightningDataModule):
    def __init__(self, config, data_utils:dataset_utils, proc:Wav2Vec2Processor) -> None:
        super().__init__()
        self.config = config
        self.cores = config.cores
        self.batch_size = config.batch_size
        self.data_utils = data_utils
        self.train = audio_dataset.dataset(self.data_utils.train, self.config.audio_path, proc)
        self.test = audio_dataset.dataset(self.data_utils.test, self.config.audio_path, proc)
        self.val = audio_dataset.dataset(self.data_utils.validate, self.config.audio_path, proc)
        

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.cores, collate_fn=self.train.collocate)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.cores, collate_fn=self.val.collocate)
            
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.cores, collate_fn=self.test.collocate)






