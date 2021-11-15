import argparse
from genericpath import isdir
from posix import listdir
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.plugins import DDPPlugin

from torch.utils import data
import models
import model_utils
from os import listdir as ls
from os.path import isdir
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

#
def main(args):
    wav2vec = Wav2Vec2ForCTC.from_pretrained(args.pretrained if args.pretrained in ls() else f'facebook/{args.pretrained}')
    processor = Wav2Vec2Processor.from_pretrained(args.pretrained+'_proc' if args.pretrained+'_proc' in ls() else f'facebook/{args.pretrained}')
    if args.pretrained not in ls(): ##save locally for fasterloading
        wav2vec.save_pretrained(args.pretrained)
        processor.save_pretrained(args.pretrained+'_proc')
    
    wav2vec.freeze_feature_extractor()
 
    data_util = model_utils.dataset_utils(args)
    if args.train_csv_path == None:
        data_util.create_splits()
    else:
        data_util.load_datasets(args)
   
    model = models.ASR_CTC(wav2vec, args)

    data_module = model_utils.DataModule(args, data_util, processor)
    checkpoint = ModelCheckpoint(dirpath='checkpoints/', monitor='val_loss', mode='min')

    trainer = pl.Trainer(gpus=args.gpus, 
        strategy='ddp', 
        precision=16, 
        max_epochs=args.epochs, 
        min_epochs=1, 
        callbacks=[checkpoint], 
        progress_bar_refresh_rate=10, 
        log_every_n_steps=3
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', help='path to pretrained model', default='wav2vec2-large-robust-ft-libri-960h')
    parser.add_argument('--data_csv_path', help='path to entire csv file with data', default='data/line_index.csv')
    parser.add_argument('--train_csv_path', help='path to training csv file with data', default='train.csv')
    parser.add_argument('--test_csv_path', help='path to testing csv file with data', default='test.csv')
    parser.add_argument('--val_csv_path', help='path to validation csv file with data', default='val.csv')
    parser.add_argument('--data_path', help='path to folder to store data csv', default='data/')
    parser.add_argument('--audio_path', help='path to folder containing wav files that are referenced in csv', default='data/audio')
    parser.add_argument('--gpus', help='number of gpus to use', default=-1, type=int)
    parser.add_argument('--epochs', help='number of epochs to train', default=10, type=int)
    parser.add_argument('--batch_size', help='batch size', default=1, type=int)
    parser.add_argument('--learning_rate', '-lr', help='learning rate', default=1e-15, type=float)
    parser.add_argument('--cores', help='number of cores to use', default=1, type=int)

    args = parser.parse_args()
    torch.cuda.empty_cache()
    main(args)
