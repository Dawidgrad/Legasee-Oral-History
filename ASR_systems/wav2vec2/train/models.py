import torch
import torch.nn as nn
import pytorch_lightning as pl 
#from deepspeed.ops.adam import FusedAdam
#from deepspeed.ops.adam.cpu_adam import CPUAdamBuilder

class ASR_CTC(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, batch):
        return self.model(batch['input_values']) if 'input_ids' not in batch else self.model(batch['input_values'], labels=batch['input_ids'])

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return self.optimizer

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def batch_dict(self, batch):
        return {
            'input_values':batch[0],
            'input_ids':batch[1]
        }

    def training_step(self, batch, batch_idx):
        loss = self.forward(self.batch_dict(batch)).loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, sync_dist=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.forward(self.batch_dict(batch)).loss
        self.log('test_loss', loss, prog_bar=True, on_step=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(self.batch_dict(batch)).loss
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return {'val_loss': loss}

