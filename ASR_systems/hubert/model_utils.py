from ast import Return
import numpy as np
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
import torch
import pandas as pd
from transformers import Wav2Vec2Processor
import audio_dataset
from torch.utils.data.distributed import DistributedSampler
import logging
import torch.distributed as dist
from typing import List, Dict
from tqdm import tqdm
from os.path import join

from os import listdir

from deepspeed.ops.adam import FusedAdam
import bitsandbytes as bnb
from adafactor import Adafactor

from torch.distributed.optim import ZeroRedundancyOptimizer as Zero
from torch.distributed.algorithms.join import Join

from os import mkdir
from os import rmdir

from torch.cuda.amp import GradScaler, autocast
import wandb
def isthere(config, element:str):
    return True if element in config else False


def load_datasets(config, processor):
    if isthere(config, 'data_csv_path'):
        master = pd.read_csv(config.data_csv_path)
        train = master.loc[master['split'] == 'train']
        validate = master.loc[master['split'] == 'val']
        train_dataset = audio_dataset.dataset(train, config.audio_path, processor)
        validate_dataset = audio_dataset.dataset(validate, config.audio_path, processor)
        return train_dataset, validate_dataset
    else:
        Exception('No data_csv_path specified')


def batch_to_device(batch, device):
    return {
        'input_values': batch['input_values'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'input_ids': batch['input_ids'].to(device),
    }


def save_model(args, model, optim, epoch, val_loss, rank):
    optim.consolidate_state_dict(to=0) # consolidate optimizer states on rank zero
    
    if rank == 0:
        print('Saving model...')
        unique_name = np.random.randint(0, 100000)
        save_dir = join(args.save_dir, f'{unique_name}_{epoch}')
        mkdir(save_dir)
        model.model.save_pretrained(join(save_dir, 'model'))
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'optimizer': optim.state_dict(),
            'config': args,
        }, join(save_dir, 'checkpoint.pt'))

        with open(join(save_dir, 'val_loss.txt'), 'w') as f:
            f.write(str(val_loss))

        if len(listdir(args.save_dir)) > args.max_saves:
            oldest_save = min(listdir(args.save_dir), key=lambda x: int(x.split('_')[1]))
            rmdir(join(args.save_dir, oldest_save))
        
        

def load_optimizer(args, model):
    '''
    Zero-redundancy optimizer partitions optimizer states so that each GPU is only responsible for a its own subset of the parameters.
    Adafactor uses less memory than Adam!
    ''' 
    return Zero(model.parameters(), optimizer_class=Adafactor, lr=args.learning_rate)

def load_dataloaders(args, rank:int, train:audio_dataset.dataset, val:audio_dataset.dataset) -> Dict[str, DataLoader]:
    return {
        'train': DataLoader(
            train,
            batch_size=args.batch_size,
            collate_fn=train.collocate,
            pin_memory=False,
            sampler=DistributedSampler(
                train,
                num_replicas=args.world_size,
                rank=rank,
                shuffle=True
            )
        ),
        'val': DataLoader(
            val,
            batch_size=args.batch_size,
            collate_fn=train.collocate,
            pin_memory=False,
            sampler=DistributedSampler(
                val,
                num_replicas=args.world_size,
                rank=rank,
                shuffle=True
            )
        ),
    }

def log(name, value, rank:int, pbar=None):
    if rank == 0: # Only log on rank 0
        wandb.log({name: value}) 
        if pbar is not None: 
            pbar.set_description(f'{name}: {value:.4f}')


def train_epoch(args, model, optimizer, dataloader, device, grad_scaler): 
    model.train()

    #with Join([model, optimizer]):
    pbar = tqdm(dataloader, desc='Train')
    train_loss_accum = 0
    for i, batch in enumerate(pbar):
        with autocast():
            loss = model(batch_to_device(batch, device))
            loss = loss / args.accumulate_grad_batches
        
        grad_scaler.scale(loss).backward()
        train_loss_accum += loss.item()

        if (i+1) % args.accumulate_grad_batches == 0: 
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

            log('train_loss', train_loss_accum, device, pbar)
            train_loss_accum = 0


def val_epoch(args, model, dataloader, device):
    model.eval()
    val_loss = 0
    n_steps = 0
    pbar = tqdm(dataloader, desc='Val')
    with torch.no_grad():
        for batch in tqdm(pbar):
            loss = model(batch_to_device(batch, device))
            val_loss += loss.item()
            n_steps += 1

    loss = val_loss / n_steps
    log('val_loss', loss, device, pbar)
    return loss

