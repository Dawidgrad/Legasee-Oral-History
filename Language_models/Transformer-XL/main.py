import argparse
import torch
from torch.utils.data import dataloader
import model_utils
import data_utils
from os import listdir as ls

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = data_utils.get_dataloader(ls(args.train_dir), data_utils.tokenize, args.batch_size, args.train_dir, True, 1)
    print('--- Data loaded ---')
    if args.ckpt == False:
        model = model_utils.return_untrained_model()
        model.to(device)
        print('--- Model Loaded ---')
        trainer = model_utils.Trainer(model=model, epochs=args.epochs, trainloader=dataloader, batch_chunks=args.batch_chunks, lr=args.lr)
        print('--- Trainer Loaded ---')
        print('Total Number of parameters: ', trainer.model.num_parameters())
        trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_chunks", type=int, default=1000, help="size of chunks to split batch sequences into")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size') #note only working for batch size of 1 atm because of bug with hugging face
    parser.add_argument('--lr', type=float, default=0.00025, help="learning rate")
    parser.add_argument('--ckpt', type=bool, default=False, help="whether to load from a checkpoint")
    parser.add_argument('--ckpt_path', type=str, default="", help="path to checkpoint")
    parser.add_argument('--train_dir', type=str, default="./data/", help="path to training data")
    args = parser.parse_args()
    main(args)
