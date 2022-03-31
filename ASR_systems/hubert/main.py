import argparse
import os
from torch.utils import data
import models
import model_utils
from os import listdir as ls
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Processor, HubertForCTC

from torch.utils.checkpoint import checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch.cuda.amp import GradScaler

def setup(rank, worldsize):
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def get_pretrained(args, gpu):
    hubert = HubertForCTC.from_pretrained(args.pretrained )
    processor = Wav2Vec2Processor.from_pretrained(args.pretrained+'_proc')
 
    model = models.ASR_CTC(hubert, args)
    if args.ckpt != 'NONE':
        model = model_utils.load_model(args, model, gpu)
    model.model.freeze_feature_extractor()
    model.model.config.gradient_checkpointing = True
    print('--- Loaded, and Feature Extractor Frozen ---')

    return model, processor
    
def run(args, model, loaders, rank):
    torch.cuda.set_device(rank)
    if rank == 0:
        #print(model)
        pass 

    model = DDP(model.to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True, gradient_as_bucket_view=True)
    print('--- DDP Model initialized ---')
    optim = model_utils.load_optimizer(args, model)
    print('--- Optimizer initialized ---')
    grad_scaler = GradScaler() 
    min_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'--- Epoch {epoch+1} ---')
        model_utils.train_epoch(args, model, optim, loaders['train'], rank, grad_scaler)
        val_loss = model_utils.val_epoch(args, model, loaders['val'], rank)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model_utils.save_model(args, model, optim, epoch, val_loss, rank)
        print(f'--- Epoch {epoch+1} Complete ---')


def main(gpu, args):
    if args.world_size > 1:
        setup(gpu, args.world_size)
    else:
        print('--- This code is only written for distributed training due to the large model size ---')


    if gpu == 0:
        if args.wandb_resume != "":
            wandb.init(id=args.wandb_resume, project='Wav2vec2', resume='must', entity='slt-cdt-team-a')
            print('--- Wandb Resumed ---')
        else:
            wandb.init(project="Wav2vec2", entity='slt-cdt-team-a')
            print(f'--- Wandb Initialized as {wandb.run.id} ---')

    hubert, processor = get_pretrained(args, gpu)
    train, val = model_utils.load_datasets(args, processor)
    print('--- Datasets loaded ---')
    dataloaders = model_utils.load_dataloaders(args, gpu, train, val)
    
    run(args=args, model=hubert, loaders=dataloaders, rank=gpu)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs')
    #parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
    parser.add_argument('--pretrained', help='path to pretrained model', default='/home/acp21rjf/Legasee-Oral-History/ASR_systems/wav2vec2/train/hubert-xlarge-ls960-ft')
    parser.add_argument('--ckpt', help='path to trained model as checkpoint', default='NONE')
    parser.add_argument('--data_csv_path', help='path to entire csv file with data', default='./data/hubert_train.csv')
    
    parser.add_argument('--accumulate_grad_batches', help='accumulate gradients over batches', default=16, type=int)
    parser.add_argument('--data_path', help='path to folder to store data csv', default='./data/')
    parser.add_argument('--audio_path', help='path to folder containing wav files that are referenced in csv', default='./data/')
   
    parser.add_argument('--epochs', help='number of epochs to train', default=30, type=int)
    parser.add_argument('--batch_size', help='batch size', default=4, type=int) 
	
    parser.add_argument('--learning_rate', '-lr', help='learning rate', default=1e-4, type=float) # base learning rate
    parser.add_argument('--max_lr', help='max learning rate', default=1.2e-4, type=float)   
    parser.add_argument('--cycle_len', help='cycle length', default=200, type=int) # step size up to max lr (step size down = 1)

    parser.add_argument('--save_dir', help='path to save model', default='/home/acp21rjf/Legasee-Oral-History/ASR_systems/wav2vec2/train/hubert_train/checkpoints/')
    parser.add_argument('--max_saves', help='max number of saves', default=1, type=int)

    parser.add_argument('--cores', help='number of cores to use', default=0, type=int)
    parser.add_argument('--wandb', help='whether to use wandb', default=True, type=bool)    

    parser.add_argument('--wandb_resume', help='whether to resume wandb "" = don\'t ', default="" , type=str)
    

    args = parser.parse_args()
    world_size = os.environ['WORLD_SIZE'] 
    print(f'-- starting for {world_size} gpus --') 

    local_rank = int(os.environ["LOCAL_RANK"])

    main(gpu=local_rank, args=args)
    """
    if args.world_size == 1:
        main(gpu=0, args=args)
    else:
        mp.spawn(main, args=(args,), nprocs=args.world_size)
    """

