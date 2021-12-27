from typing import Dict
import numpy as np
import torch
from tqdm import tqdm

# TODO :

# Store discarded beams, so that they can be merged if necessary

# ------------ #
# Optimization #
# ------------ #

# dictionary lookup is 6.6 times faster than a list, so replace all lists with dicts where possible
# Remove redundant looping of lists
# Improve beam collapsing 
# Replace recursive calls with a loop                                                                        ✓✓ Done ✓✓
# After certain sequence length store LM models hidden states in memory and use them for the next sequence
# Replace looped multiplication with matrix operations where possible 
# cache beam strings 
# work in log space where possible, addition > multiplication                                                ✓✓ Done ✓✓

# https://stackoverflow.com/questions/11232597/storing-python-objects-in-a-python-list-vs-a-fixed-length-numpy-array

# Convert algorithm to consist of a dictionary of sequences, with the keys being the indices of <
# a matrix of corresponding log probabilities. Then batches, beam merging and score updates can <
# organized as matrix addition and logsumexp operations. Indices can be selected using indexing <
# use numpy.fromfunction to create the matrix of indices. This is more parallelizable ######### <

# https://arxiv.org/pdf/1902.06022.pdf ?

class Beam():
    '''
    Manages each instance of a beam
    '''
    def __init__(self, seq, score, pad_tkn=0, space_tkn=29):
        self.seq = seq
        self.score = score
        self.pad_tkn = pad_tkn
        self.space_tkn = space_tkn
        self.next_logits = None

    def collapse(self): # This should be optimized to avoid collapsing parts of the sequence that have already been collapsed
        seq = []
        for i, tkn in enumerate(self.seq):   # This doesn't account for spaces with a padding token between them # Fixed
            if len(seq) == 0: # first token in sequence cannot be a space or pad token
                if tkn != self.space_tkn and tkn != self.pad_tkn:   
                    seq.append(tkn)
            elif tkn != seq[-1]: #if the current token is not the same as the previous token then add it to the sequence
                if tkn == self.space_tkn and i > 1 and self.seq[i-1] == self.pad_tkn and self.seq[i-2] == self.space_tkn:
                    pass # this fixes the double space issue.. probably a better way to do all this
                else: 
                    seq.append(tkn)
        self.seq = seq

    def __str__(self):
        '''
        Returns a string representation of the beam
        '''
        return "".join(map(str, self.seq))

    def __len__(self):
        return len(self.seq)

    def get_seq(self):
        return self.seq

class BeamSearch():
    '''
    - CTC beam search algorithm
    - Vocab for LM must be the same as vocab for ASR
    - Beam_width must be computatable as one batch of inputs to the LM model
    '''
    def __init__(self, beam_width:int, asr_logits:np.array, vocab:Dict, lm_model, lm_weight=0.5, pad_tkn=0, space_tkn=29):
        self.beam_width = beam_width # max number of hypotheses to be considered at each time step
        self.vocab = vocab 
        self.reverse_vocab = {v:k for k,v in vocab.items()} 
        self.asr_logits = asr_logits # Wav2vec2 logits [seq_len, vocab_size]
        self.pad_tkn = pad_tkn       # token used for padding
        self.space_tkn = space_tkn   # token used for space
        self.lm = lm_model           # LM model __call__ must return log softmax over vocab [vocab_size]
        self.beams = []              # list of all beam objects
        self.lm_weight = -np.inf if lm_weight == 0 else np.log(lm_weight)   # weight for language model
        self.asr_weight = np.log(1-lm_weight) # weight for acoustic model

        self.current_pos = 0         # current position of the search within the logits


        self.cache = {}              # cache for lm logits that have already been computed
        self.debug = False         

        if lm_weight == 0: 
            self.lm = DummyLM()      # if lm_weight is 0 then perform greedy search

    def _get_sorted(self, current_pos): # current_pos is a bad variable name, as self.current_pos is the current position in the logits, is confusing
        '''
        Returns array where the index corresponds to the order of sorted logits i.e [0] = most probable, [-1] = least probable
        and the value is the number corresponding to a character in the vocabulary 
        '''
        sorted_ = (-current_pos).argsort() 
        beam_order = np.empty(sorted_.shape,dtype=np.int8) 
        for i, el in enumerate(sorted_): 
            beam_order[el] = i
        return beam_order

    def _init_beams(self):
        beam_order =  self._get_sorted(self.asr_logits[self.current_pos]) 
        for char in beam_order:
            if len(self.beams) >= self.beam_width:
                break
            self.beams.append(Beam([char], self.asr_logits[self.current_pos, char]))
        for beam in self.beams:
            beam.collapse()
            

    def _convert_to_text(self, seq):
        return "".join([self.reverse_vocab[el] for el in seq if el != self.pad_tkn])

    def _return_beams(self):
        return [( self._convert_to_text(beam.get_seq()), beam.score ) for beam in self.beams]
    

    def _get_beam_str(self, beam):
        '''
        String representation of the beam, ignoring the pad tokens
        '''
        beam_seq = beam.get_seq() 
        return "".join([str(el) for el in beam_seq if el != self.pad_tkn]) #string identifier for the beam, padding tokens are not included in the identifier
    
    def _merge_beams(self, beams): 
        '''
        Merge identical beams
        '''
        cbeams = [] # list to hold the merged beams
        blist = []  # list to hold string identifiers for the beams that have already been stored
        for beam in beams:
            beam_str = self._get_beam_str(beam)
            if beam_str not in blist:
                cbeams.append(beam)
                blist.append(beam_str)
            else:
                cbeams[blist.index(beam_str)].score = np.logaddexp(beam.score, cbeams[blist.index(beam_str)].score) #add the score of the new beam to the existing beam
        return cbeams


    def _retrieve_batch(self):
        '''
        Updates the next_logits for each beam in the beam list
        '''
        def pad_batch(seq): # pad the batch to max seq length and returns a mask to be passed to the model
            max_len = max([len(el) for el in seq])
            tgt_indices = torch.empty(len(seq), dtype=torch.long) # target indices to select lm predictions from i.e we want the last token that isn't padding
            for i, el in enumerate(seq):
                tgt_indices[i] = len(el) - 1
                el.extend([self.pad_tkn]*(max_len-len(el)))
            return torch.tensor(seq, dtype=torch.long), tgt_indices 

        sequences = []      # store sequences to be passed to the LM
        beam_indices = []   # indices of the sequences corresponding to the beam
        batch_indices = []  # indices of the sequences corresponding to the batch
        for i, beam in enumerate(self.beams):
            cur_seq = beam.get_seq() 
            cur_seq = [el for el in cur_seq if el != self.pad_tkn] # remove padding tokens from the sequence, before passing to the LM
            cur_seq_str = self._get_beam_str(beam) # change this maybe as this is performing a similar function to the above

            if len(cur_seq) == 0:          # hmm should add start token to lm model during training for this
                cur_seq = [self.space_tkn] # if the sequence is empty then add a space token
                batch_indices.append(len(sequences))
                sequences.append(cur_seq)
                beam_indices.append(i)

            elif cur_seq_str not in self.cache:
                batch_indices.append(len(sequences))
                sequences.append(cur_seq)
                beam_indices.append(i)

            else:
                print('cache hit') if self.debug else None
                beam.next_logits = self.cache[cur_seq_str]

        if len(sequences) != 0:
            sequences, tgt_indices = pad_batch(sequences)
            batch_logits = self.lm(sequences, tgt_indices) 

            for i_beam, i_batch in zip(beam_indices, batch_indices):        
                self.beams[i_beam].next_logits = batch_logits[i_batch]      # update the next_logits for each beam
                self.cache[str(self.beams[i_beam])] = batch_logits[i_batch] # cache the logits

     


    def _get_candidate(self, beam:Beam):
        '''
        Returns list of candidate next beams. Every item in the vocabulary is a potential candidate for the search
        '''
        if beam.next_logits is None:
            print('Next logits is None, this should not happen')
            return [Beam([*beam.get_seq(), token], beam.score + self.asr_logits[self.current_pos, token]) for token in self.vocab.values()]
        else:
            return [Beam([*beam.get_seq(),token], beam.score + ( np.logaddexp(self.lm_weight+beam.next_logits[token], self.asr_weight+self.asr_logits[self.current_pos, token]) ) ) for token in self.vocab.values()] #compute the score for each candidate beam, combining the score from the language model and the score from the asr model

    def _search_step(self):
        candidate_sets = [self._get_candidate(beam) for beam in self.beams]
        candidate_beams = [item for sublist in candidate_sets for item in sublist] #unpack the candidate sets
        for beam in candidate_beams:
            beam.collapse()
        candidate_beams = self._merge_beams(candidate_beams)
        candidate_beams.sort(key=lambda beam: beam.score, reverse=True) #sort by score (largest to smallest)
      
        self.beams = candidate_beams[:self.beam_width] #prune to beam_width


    def search(self): #main search function that returns the top k beams
        for i in tqdm(range(len(self.asr_logits)), desc='Searching'):
          self.current_pos = i
          ### End of sequence 
          if self.current_pos == len(self.asr_logits) - 1: 
              return [self, self._return_beams()] if self.debug else self._return_beams()   
          ## Start of sequence
          elif len(self.beams) == 0:
              self._init_beams() 
          # During sequence
          else:
              self._search_step() 

          self._retrieve_batch() #retrieve logits for next character for each beam
    
        
    

            
class DummyLM():
    '''
    Dummy Language Model for greedy search when LM weight is 0
    '''
    def __init__(self, vocab_size=30):
        self.vocab_size = vocab_size
    def __call__(self, batch:torch.tensor, indices:torch.tensor):
        out = np.ones((batch.shape[0], self.vocab_size))
        return out

