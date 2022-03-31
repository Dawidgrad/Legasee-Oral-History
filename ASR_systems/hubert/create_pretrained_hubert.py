import argparse
import torch
from transformers import HubertForCTC
from models import ASR_CTC

def convert_from_ddp(model_state_dict):
    '''
    Convert model state dict from DDP to single GPU.
    '''
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


PATH = './checkpoints/hubert_all_21/checkpoint.pt'
MODEL = '../hubert-xlarge-ls960-ft'
OUTF = './hubert_all'

hubert = HubertForCTC.from_pretrained(MODEL)
model = ASR_CTC(hubert)
weights = convert_from_ddp(torch.load(PATH)['model'])
model.load_state_dict(weights)
model.model.save_pretrained(OUTF)

print('--- SAVE COMPLETE ---')
