from ast import arg
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from parallelformers import parallelize
import pickle as pkl
import multiprocessing
import kenlm
from pyctcdecode import build_ctcdecoder
import multiLM
from Levenshtein import distance as levenshtein_distance
from typing import List, Tuple

def logging(path, data):
    if path != "":
        with open(path, 'a') as f:
            f.write(data + '\n')


def load_model(args):
    model = HubertForCTC.from_pretrained(args.model) if args.hubert == True else Wav2Vec2ForCTC.from_pretrained(args.model)
    processor = Wav2Vec2Processor.from_pretrained(args.processor)
    model.eval()
    if args.gpus > 0:
        if args.confidence == 0:
            parallelize(model, num_gpus=args.gpus, fp16=args.fp16) # model parallelization
        elif args.gpus == 1:
            model.to(torch.device('cuda'))
        else:
            model = torch.nn.DataParallel(model) # implement for distributed
            model.to(torch.device('cuda'))   
    else:
        model.to('cpu') 
    return model, processor

def stack_batch(batch:List[torch.Tensor], batch_size:int) -> List[torch.Tensor]:
    '''
    Input a list of input tensors and a batch size -> return a list of tensors stacked by batch size
    '''
    return [torch.stack(batch[i:i+batch_size]) for i in range(0, len(batch), batch_size)]
    

def get_hypotheses(args, model, inputs, k=10) -> List[torch.Tensor]:
    '''
    Returns outputted logits from k forward passes with dropout enabled for confidence estimation
    '''
    enable_dropout(model) # enable dropout layers
    with torch.no_grad():
        batch = stack_batch([inputs for _ in range(args.confidence)], args.batch_size)
        hypotheses = [model(**el).logits.cpu() for el in batch] 
    hypotheses = torch.stack(hypotheses, dim=0).argmax(dim=-1)
    model.eval() # disable dropout layers
    return hypotheses

def get_confidence(proc:Wav2Vec2Processor, reference:torch.Tensor, hypothesis:torch.Tensor):
    '''
    Calculates the confidence of the model.
    '''
    all_max_ids = torch.cat([reference.unsqueeze(0), hypothesis])
    text_outputs = proc.batch_decode(all_max_ids)
    ref, hyp = text_outputs[:1][0], text_outputs[1:]
    conf = get_levenstein_batch(hyp, ref)
    return conf

def get_levenstein_batch(labels, gold):
    '''
    Gets the max of the levenstein distance (normalized by the length of the label) between the predictions and the gold (gold = deterministic forward pass)
    '''
    lev = [levenshtein_distance(gold, label) for label in labels]
    # normalize based on label length
    levs = [lev[i] / len(gold) for i in range(len(labels))]
    max_l = max(levs)
    return max_l

def enable_dropout(model:Wav2Vec2ForCTC):
    '''sets dropout layers to train'''
    num = 0
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            num += 1
    print(f'{num} dropout layers enabled')


def apply_softmax(logits:torch.tensor) -> torch.tensor:
    with torch.no_grad():
        smax = torch.nn.LogSoftmax(dim=-1)
        lpz = smax(logits).numpy()
    return lpz


def __kenlm_decoder(arpa, vocab, alpha=0.5, beta=0.8):  
    alpha = alpha
    beta = beta
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    return decoder


def load_decoder(args, vocab):
    if args.kenlm != '':
        if args.kenlm2 == '':
            decoder = __kenlm_decoder(args.kenlm, vocab, args.alpha, args.beta)
            print(f'--- LM loaded: {args.kenlm} ---')
        else:
            decoder = multiLM.build_ctcdecoder(multiLM.get_vocab(), [args.kenlm, args.kenlm2], alpha=args.alpha, beta=args.beta)
            print(f'--- LM loaded: {args.kenlm}, {args.kenlm2} ---')
    else:
        decoder = __kenlm_decoder(None, vocab, args.alpha, args.beta)
    return decoder

def get_dict_index(dict, index):
    return {key: dict[key][index] for key in dict}


def run_model(args, model, processor, chunks, batch_size):
    out_lst = []
    conf_lst = []

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            inp = processor(batch, padding='longest', return_tensors='pt', sampling_rate=16000)
            out = model(**inp).logits.cpu()
    
            if args.confidence != 0:
                for ix in range(len(batch)):
                    hypoth = get_hypotheses(args, model=model, inputs=get_dict_index(inp, ix), k=args.confidence)
                    ref = out[ix].reshape(-1, 32).argmax(dim=-1)
                    conf = get_confidence(processor, ref, hypoth)
                    conf_lst.append(conf)

            logging(args.log_pth, f'Batch {i}-{i+batch_size} of {len(chunks)} -- {out.shape}')
            out_lst.append(out.numpy())


    return out_lst, conf_lst

### word localization

def get_b_time(beam_stamps, index_duration, start_t):
  return [{'text':el[0], 'start':el[1][0]*index_duration+start_t, 'end':el[1][1]*index_duration+start_t} for el in beam_stamps]

def process_beams(beam_list, chunks, chunk_idxs, logits):
  out_lst = []
  cur_time = 0
  for i, beam in enumerate(beam_list):
    cur_time = chunk_idxs[i]['start']
    topbeam = beam[0][1]
    index_duration = chunks[i].shape[0] / logits[i].shape[0] / 16000
    out_lst.extend(get_b_time(topbeam, index_duration, cur_time))
  return out_lst

### word localization

def decode_lm(args, logits, decoder, chunks, chunk_idxs):
    ''' decode logits to text '''
    beam_width = args.beam_width if args.kenlm != '' else 1
    logging(args.log_pth, f'--- Decoding LM with beam width {beam_width} ---')
    # unpack logits insto list of [-1, 32] tensors
    logit_list = []
    for i in range(len(logits)):
        for z in range(len(logits[i])):
            logit_list.append(logits[i][z])

    with multiprocessing.get_context('fork').Pool() as pool:
        decoded = decoder.decode_beams_batch(pool, logit_list, beam_width=beam_width)

    timestamped_decoded = process_beams(decoded, chunks, chunk_idxs, logit_list)
    return timestamped_decoded


