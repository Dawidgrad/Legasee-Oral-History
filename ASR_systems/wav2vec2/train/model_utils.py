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

    def load_datasets(self): 
        if isthere(self.config, 'data_csv_path'):
            master = pd.read_csv(self.config.data_csv_path)
            self.train = master.loc[master['split'] == 'train']
            self.validate = master.loc[master['split'] == 'val']
        else:
            print("Please specify 'data_csv_path' in args")
             
      

class DataModule(pl.LightningDataModule):
    def __init__(self, config, data_utils:dataset_utils, proc:Wav2Vec2Processor) -> None:
        super().__init__()
        self.config = config
        self.cores = config.cores
        self.batch_size = config.batch_size
        self.data_utils = data_utils
        self.train = audio_dataset.dataset(self.data_utils.train, self.config.audio_path, proc)
       
        self.val = audio_dataset.dataset(self.data_utils.validate, self.config.audio_path, proc)
        

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.cores, collate_fn=self.train.collocate)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.cores, collate_fn=self.val.collocate)
    






