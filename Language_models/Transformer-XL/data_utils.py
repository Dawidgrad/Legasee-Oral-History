import torch
from torch.utils.data import Dataset, DataLoader
from os import listdir as ls
import numpy as np
import re

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
}

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

class DatasetXL(Dataset):
    '''
    Dataset for language modeling with TransformerXL - each batch should contain one entire continous sequence 
    '''
    def __init__(self, folders:list, tokenizer, train_dir:str):
        self.folders = folders
        self.tokenizer = tokenizer
        self.train_dir = train_dir
        self.data = self.load()
        print(f'Number of Training Documents: {len(self)}')

    def collate_fn(self, batch):
        '''
        Collate function for the dataloader
        '''
        maxl = max([len(el) for el in batch])
        padded = np.array([np.pad(el, (0, maxl - len(el)), 'constant', constant_values=0) for el in batch])
        
        return torch.tensor(padded, dtype=torch.long)

    def load(self) -> list:
        '''
        Load the data from the folders
        '''
        #print(self.folders[0])
        data = []
        ttl_len = 0
        for folder in self.folders:
            for file in ls(self.train_dir+folder): 
                if(file[-3:] == 'txt'): #check if file is a text file
                    with open(self.train_dir+folder + '/' + file, 'r') as f:
                        dta = self.tokenizer(f.read())
                        ttl_len += len(dta)
                        data.append(dta)
        print(f'Total Characters: {ttl_len}')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(folders:list, tokenizer,  batch_size:int, train_dir:str, shuffle:bool, num_workers:int):
    '''
    Get dataloader for the dataset
    '''
    dataset = DatasetXL(folders, tokenizer, train_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate_fn)
