import torch
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLConfig
from tqdm import tqdm

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
    model = TransfoXLLMHeadModel(config)
    return model


class Trainer():
    def __init__(self, model:TransfoXLLMHeadModel, epochs:int, trainloader:torch.utils.data.DataLoader, batch_chunks:int, lr:float):
      self.model = model
      self.epochs = epochs
      self.trainloader = trainloader
      self.b_chunks = batch_chunks
      self.lr = lr
      self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.losses = []

    def onTrainEnd(self):
        print('--- Saving model ---')
        self.model.save_pretrained('Transformer-XL')
        print('--- Model saved ---')

    def fit(self):
        for epoch in range(self.epochs):
            print(f'--- Epoch {epoch+1}/{self.epochs} ---')
            pbar = tqdm(self.trainloader)
            for batch, data in enumerate(pbar):
                mems = None #init mems as None
                input_chunks = torch.split(data, self.b_chunks, 1)
                tgt_chunks = torch.split(data, self.b_chunks, 1)
                losses = []
                #print(len(input_chunks))
                for i in range(len(input_chunks)):
                    inp_i = input_chunks[i].contiguous()
                    tgt_i = tgt_chunks[i].contiguous()

                    #print(inp_i.size(1))
                    if inp_i.size(1) == 0:
                      continue #skip if empty
                      
                    self.optim.zero_grad() #clear gradients
                    out = self.model(inp_i.to(self.device), labels=tgt_i.to(self.device), mems=mems)
                    loss, _, mem = out[0], out[1], out[2]
                    mems = mem
                    loss = loss.mean()
                    loss.backward()
                    self.optim.step()
                    losses.append(float(loss.detach().cpu().numpy()))
                if(len(losses)!=0):
                  mean_loss = sum(losses)/len(losses)
                  pbar.set_description(f'loss: {mean_loss}')
                  self.losses.append(mean_loss)
        print('--- Training finished ---')
        self.onTrainEnd()
        

