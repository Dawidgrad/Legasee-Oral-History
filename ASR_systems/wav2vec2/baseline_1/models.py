import torch
import torch.nn as nn

class ASR_CTC(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.device_ = "cuda" if torch.cuda.is_available() else 'cpu'

    def forward(self, batch, labels=None):
        return self.model(batch.input_values) if labels == None else self.model(batch['input_values'], labels=labels)
