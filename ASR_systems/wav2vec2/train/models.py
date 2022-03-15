import torch
import torch.nn as nn
import pytorch_lightning as pl 
import bitsandbytes as bnb
from torch.optim.lr_scheduler import LambdaLR

class ASR_CTC(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, batch):
        return self.model(batch['input_values']) if 'input_ids' not in batch else \
            self.model(batch['input_values'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])

    def configure_optimizers(self):
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.optimizer = bnb.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate) # 8-bit AdamW for reduced memory usage
        # linear warmup schedule
        warmup = lambda step: min(step / self.config.warmup_steps, 1.0)
        self.schedular = LambdaLR(self.optimizer, lr_lambda=warmup)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.schedular,
                'interval': 'step',
            }
        }

    def batch_dict(self, batch):
        return {
            'input_values':batch[0],
            'attention_mask':batch[1],
            'input_ids':batch[2]
        }

    def training_step(self, batch, batch_idx):
        loss = self.forward(self.batch_dict(batch)).loss 

        self.log('train_loss', loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log('lr', self.schedular.get_last_lr()[0], prog_bar=True, on_step=True, sync_dist=True)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.forward(self.batch_dict(batch)).loss
        self.log('test_loss', loss, prog_bar=True, on_step=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(self.batch_dict(batch)).loss 
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return {'val_loss': loss}

