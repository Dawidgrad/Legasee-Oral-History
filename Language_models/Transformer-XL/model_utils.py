from typing import Dict
import torch
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLConfig
from tqdm import tqdm
import transformers
import subprocess as sp
import os

def get_gpu_memory(): # https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values



config = TransfoXLConfig(
    d_embed=256,
    d_model=256,
    d_head=32,
    n_head=8,
    n_layer=12,
    vocab_size=30,
    d_inner=2048,
    cutoffs=[],
    adaptive=False,
)



def return_untrained_model():
    # set model.crit to -> torch.nn.CrossEntropyLoss() to allow for masked loss w/ -100 as padding
    # no need for adaptive softmax w/ vocab_size=30
    model = TransfoXLLMHeadModel(config)
    model.crit = torch.nn.CrossEntropyLoss()
    model.out_layer = torch.nn.Linear(config.d_model, config.vocab_size)
    return model


class Trainer():
    def __init__(self, model:TransfoXLLMHeadModel, epochs:int, dataloaders:Dict, batch_chunks:int, accum:int, lr:float, save_folder:str='./chechkpoints'):
      self.model = model
      self.epochs = epochs
      
      self.trainloader = dataloaders['train']
      self.valloader = dataloaders['val']
      self.testloader = dataloaders['test']

      self.b_chunks = batch_chunks
      self.lr = lr
      self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.losses = []
      self.accum = accum


    def __forward(self, inputs, labels):
            transformer_outputs = self.model.transformer(
                inputs.to(self.device),
                mems=labels,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
            )
            last_hidden = transformer_outputs[0]
            pred_hidden = last_hidden[:, -inputs.size(1):] 
            proj_out = self.model.out_layer(pred_hidden)
            proj_out = proj_out.view(-1, self.model.config.vocab_size, inputs.size(1))
            loss = self.model.crit(proj_out, labels.to(self.device))
            mems = transformer_outputs.mems
            return loss, mems

    def __save_model(self, epoch):
        print('--- Saving model ---')
        state = {
            'epoch': str(epoch),
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }
        torch.save(state, os.path.join(self.save_folder, f'model_epoch_{state["epoch"]}.pth'))
        print('--- Model saved ---')

    def fit(self):
        for epoch in range(self.epochs): 
            print(f'--- Epoch {epoch+1}/{self.epochs} ---') 
            pbar = tqdm(self.trainloader)
            for batch, data in enumerate(pbar): 
                mems = None #init mems as None
                inputs, labels = data
                input_chunks = torch.split(inputs, self.b_chunks, 1)
                tgt_chunks = torch.split(labels, self.b_chunks, 1) 
       
                losses = []
       
                for i in range(len(input_chunks)):
                    inp_i = input_chunks[i].contiguous() 
                    tgt_i = tgt_chunks[i].contiguous()
                    if inp_i.size(1) == 0:
                      continue #skip if empty
           
                    loss, mems = self.__forward(inp_i, tgt_i)

                    loss.backward()
                    
                    if (i+1) % self.accum == 0: # accumulate gradients for BPTT # % is modulo operator, divides right by left and returns remainder
                        self.optim.step()
                        self.optim.zero_grad() #clear gradients

                    losses.append(float(loss.detach().cpu().numpy())) # just use .item() ?

                if(len(losses)!=0):
                  mean_loss = sum(losses)/len(losses)
                  pbar.set_description(f'loss: {mean_loss}')
                  self.losses.append(mean_loss)

            val_losses = []
            for batch, data in enumerate(self.valloader):
                inputs, labels = data
                input_chunks = torch.split(inputs, self.b_chunks, 1)
                tgt_chunks = torch.split(labels, self.b_chunks, 1) 
       
                for i in range(len(input_chunks)):
                    inp_i = input_chunks[i].contiguous() 
                    tgt_i = tgt_chunks[i].contiguous()
                    if inp_i.size(1) == 0:
                      continue
                    with torch.no_grad():
                        loss, mems = self.__forward(inp_i, tgt_i)
                    losses.append(float(loss.detach().cpu().numpy()))
            mean_val_loss = sum(val_losses)/len(val_losses)
            print(f'Val_loss: {mean_val_loss}')

            self.__save_model(epoch)
                  
        print('--- Training finished ---')
        self.__save_model('final')
        

