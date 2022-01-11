import argparse
from numpy import split
import torch
from torch.utils.data import dataloader
import model_utils
import data_utils
from os import listdir as ls
import pandas as pd

def main(args):
    # dataloader uses a folder of text files to create a dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_test_val = pd.read_csv(args.train_test_val)

    train_dataloader = data_utils.get_dataloader(
        csv=train_test_val, 
        tokenizer = data_utils.tokenize, 
        batch_size = args.batch_size, 
        split='train', 
        shuffle=True, 
        num_workers=1
    )
    test_dataloader = data_utils.get_dataloader(
        csv=train_test_val, 
        tokenizer = data_utils.tokenize, 
        batch_size = args.batch_size, 
        split='train', 
        shuffle=True, 
        num_workers=1
    )
    val_dataloader = data_utils.get_dataloader(
        csv=train_test_val, 
        tokenizer = data_utils.tokenize, 
        batch_size = args.batch_size, 
        split='train', 
        shuffle=True, 
        num_workers=1
    )
    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }

    print('--- Data loaded ---')
    if args.ckpt == False:
        model = model_utils.return_untrained_model()
        model.to(device)
        print('--- Model Loaded ---')
        trainer = model_utils.Trainer(model=model, epochs=args.epochs, dataloaders=dataloaders, batch_chunks=args.batch_chunks, accum=args.accum, lr=args.lr)
        print('--- Trainer Loaded ---')
        print('Total Number of parameters: ', trainer.model.num_parameters())
        trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_chunks", type=int, default=500, help="size of chunks to split batch sequences into")
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')  
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--ckpt', type=bool, default=False, help="whether to load from a checkpoint")
    parser.add_argument('--ckpt_path', type=str, default="", help="path to checkpoint")
    parser.add_argument('--accum', type=str, default=4, help='gradient accumulation')
    parser.add_argument('--data_csv', type=str, default="train_test_val.csv", help="path to data csv file")
    args = parser.parse_args()
    main(args)
