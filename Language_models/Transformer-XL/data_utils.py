import torch
from torch.utils.data import Dataset, DataLoader
from os import listdir as ls
import numpy as np
import re
from os.path import isdir, join
from os import walk
import pandas as pd

vocab = {
    "'": 24,
    '<pad>': 0,
    '<unk>': 1,
    'A': 4,
    'B': 21,
    'C': 16,
    'D': 11,
    'E': 2,
    'F': 17,
    'G': 18,
    'H': 8,
    'I': 7,
    'J': 26,
    'K': 23,
    'L': 12,
    'M': 14,
    'N': 6,
    'O': 5,
    'P': 20,
    'Q': 27,
    'R': 10,
    'S': 9,
    'T': 3,
    'U': 13,
    'V': 22,
    'W': 15,
    'X': 25,
    'Y': 19,
    'Z': 28,
    ' ': 29
} # need to add <s> and </s> tokens to the beginning and end of each sequence

reverse_vocab = {q:k for k,q in vocab.items()}

def decode(sequence):
  seq = sequence.squeeze()
  out = [reverse_vocab[int(el)] for el in seq]
  return "".join(out)



def tokenize(string:str) -> np.array:
    """
    Tokenize a string into a list of integers
    """
    proc = re.sub("{.*?}", "", string) #parse {OOV} type tokens
    proc = " ".join([el for el in proc.split(' ') if el != '']) #remove double spaces
    return np.array([vocab[char.upper()] for char in string if char.upper() in vocab]) #convert to integers

def create_train_test_split(start_folder:str, save_name:str="train_test_val.csv"):
    '''
    Creates a dataframe for train/test/val splits by walking through the directory tree, shuffling the data, and splitting 80/10/10
    '''
    files = []
    for (dirpath, dirnames, filenames) in walk(start_folder):
        files.extend([join(dirpath, f) for f in filenames if f.endswith('.txt')])
    np.random.shuffle(files)
    train_files = files[:int(0.8*len(files))]
    test_files = files[int(0.8*len(files)):]
    val_files = test_files[:int(0.5*len(test_files))]

   
    data_files = pd.DataFrame({'files':train_files, 'split':['train']*len(train_files)})
    data_files = data_files.append(pd.DataFrame({'files':test_files, 'split':['test']*len(test_files)}))
    data_files = data_files.append(pd.DataFrame({'files':val_files, 'split':['val']*len(val_files)}))
    data_files.to_csv(save_name, index=False)
    

class DatasetXL(Dataset):
    def __init__(self, csv:pd.DataFrame, tokenizer, split:str="train"):
        self.csv = csv
        self.tokenizer = tokenizer
        self.split = split
        
        self.data = self.load()
        print(f'Number of Training Documents: {len(self)}')

    def collate_fn(self, batch):
        '''
        Collate function for the dataloader
        '''
        maxl = max([len(el) for el in batch]) 
        # rather than pad we should start a new sequence
        padded = np.array([np.pad(el, (0, maxl - len(el)), 'constant', constant_values=0) for el in batch])
        labels = np.array([np.pad(el, (0, maxl - len(el)), 'constant', constant_values=-100) for el in batch]) # -100for padding on labels to ignore loss
        
        return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def load(self) -> list:
        '''
        Loads the data from the csv file
        '''
        data = []
        ttl_len = 0
        for idx, row in self.csv.iterrows():
            if row['split'] == self.split:
                with open(row['files'], 'r') as f:
                    text = f.read()
                    text = self.tokenizer(text)
                    data.append(text)
                    ttl_len += len(text)
        print(f'Total Length of {self.split} data: {ttl_len}')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(csv:pd.DataFrame, tokenizer,  batch_size:int, split:str, shuffle:bool, num_workers:int):
    '''
    Get dataloader for the dataset
    '''
    dataset = DatasetXL(csv, tokenizer=tokenizer, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate_fn)
