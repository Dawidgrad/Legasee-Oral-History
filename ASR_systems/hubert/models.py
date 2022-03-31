from typing import Dict
import torch
import torch.nn as nn


class ASR_CTC(nn.Module):
    '''
    Custom wrapper for the model.
    '''
    def __init__(self, model, config=None):
        super().__init__()
        self.model = model
        if config:
            self.config = config
        

    def forward(self, batch:Dict[str, torch.Tensor]):
        return self.model(batch['input_values']) if 'input_ids' not in batch else \
            self.model(batch['input_values'], attention_mask=batch['attention_mask'], labels=batch['input_ids']).loss


