import torch


# TransformerXL vocab
vocabXL = {
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

#vocab for wav2vec
wav2vecVocab = {
 '<pad>': 0,
 '<s>': 1,
 '</s>': 2,
 '<unk>': 3,
 ' ': 4,
 'E': 5,
 'T': 6,
 'O': 7,
 'A': 8,
 'I': 9,
 'N': 10,
 'H': 11,
 'S': 12,
 'R': 13,
 'L': 14,
 'D': 15,
 'U': 16,
 'Y': 17,
 'W': 18,
 'M': 19,
 'C': 20,
 'G': 21,
 'F': 22,
 'P': 23,
 'B': 24,
 'K': 25,
 "'": 26,
 'V': 27,
 'J': 28,
 'X': 29,
 'Q': 30,
 'Z': 31
 }


class TransfoXLPredictonHead():
  '''
    Takes in a sequence and outputs a probability distrubution over the next character
    outputs log probabilities
  '''
  def __init__(self, model):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = model.to(self.device)
    self.model.eval()
    self.weight = model.crit.out_layers[0].weight
    self.bias = model.crit.out_layers[0].bias
    self.out_projs = model.crit.out_projs[0]  
    self.compute_logit = model.crit._compute_logit

  def __call__(self, input:torch.tensor, indices:torch.tensor):
    return self.forward(input, indices)

  def forward(self, input:torch.tensor, indices:torch.tensor): #, mems=None): not added functionality for mems yet
    if(len(input.shape) == 1):
      input = input.unsqueeze(0) # add dimension if needed
    with torch.no_grad():
      out = self.model.transformer(input.to(self.device), mems=None, output_hidden_states=True).last_hidden_state
  
      out = out[torch.tensor([el for el in range(out.shape[0])]).to(self.device), indices.to(self.device)]  # select indices for token inputs i.e we don't need predictions for padding
    
      logits = self.compute_logit(out, self.weight, self.bias, self.out_projs)
      probs = torch.nn.functional.log_softmax(logits, -1).cpu().numpy()
    return probs #return smax over next char in seq


def convertdict():
  cvrtdict = {}
  for item in wav2vecVocab.items():
    k, v = item
    if k in vocabXL:
      cvrtdict[v] = vocabXL[k]
    else:
      cvrtdict[v] = None
  return cvrtdict

def convert(output):
  '''
    Converts the output of the model to match the language models vocab before softmax is applied
  '''
  vocab_convert = convertdict()
  target = torch.empty((output.shape[0], len(vocabXL.values())), dtype=torch.float32)
  for i, step in enumerate(output):
    for j, item in enumerate(step):
      if vocab_convert[j] is not None:
        target[i][vocab_convert[j]] = item
  return target

class Wav2Vec2CTCPredictionHead():
  def __init__(self, model):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = model.to(self.device)
    self.model.eval()
    self.vocab_convert = convertdict()

    
  def __call__(self):
    return self.forward()


  def _convert(self, output): #DOESN'T WORK
    '''
      Converts the output of the model to match the language models vocab before softmax is applied
    '''
    target = torch.empty((output.shape[0], len(vocabXL.values())), dtype=torch.float32)
    for i, step in enumerate(output):
      for j, item in enumerate(step):
        if self.vocab_convert[j] is not None:
          target[i][self.vocab_convert[j]] = item
    return target

  def __forward__(self, input):
    '''
      Forward pass of the model, convert the output to match the language models vocab and apply softmax
    '''
    with torch.no_grad():
      out = self.model(input.to(self.device)).squeeze().cpu().numpy()
    convert = self._convert(out)
    return torch.nn.functional.softmax(convert, -1)