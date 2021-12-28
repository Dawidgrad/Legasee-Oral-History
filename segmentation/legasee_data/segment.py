from typing import Dict, List
import ctc_segmentation
import numpy as np
import pandas as pd
from pandas.core.indexing import check_bool_indexer
import soundfile as sf
import argparse
import torch, transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
import os

SAMPLE_RATE = 16000
PADDING = 1 # padding amount, can trim files down more later, for now bigger == better 

class utterance():
  def __init__(self, word, segment):
    self.duration = segment[1] - segment[0]
    self.segment_list = [segment]
    self.word_list = [word]

  def __str__(self):
    return " ".join(self.word_list)

  def __len__(self):
    return int(self.duration)

  def min_confidence(self):
    return np.e**min(el[-1] for el in self.segment_list)

  def avg_confidence(self):
    return np.mean([np.e**el[-1] for el in self.segment_list])

  def add(self, word, segment):
    duration = segment[1] - segment[0]
    self.duration += duration
    self.segment_list.append(segment)
    self.word_list.append(word)

def get_utterances(gap, max, min, text, segments):
  utts = []
  for word, segment in zip(text, segments):
    if len(utts)==0:
      utts.append(utterance(word, segment))
    else:
      if utts[-1].duration < max and ( (segment[0] - utts[-1].segment_list[-1][1]) < gap or utts[-1].duration < min):
        utts[-1].add(word,segment)
      else:
        utts.append(utterance(word, segment))
  return utts

def open_textfile(args, filename:str) -> str:
    with open(os.path.join(args.text_files, filename), 'r') as f:
        text = f.read()
    return text

def batch_audio(args, audio:np.array) -> np.array:
    b_chunks = [*[args.chunk_size]*(audio.shape[0]//args.chunk_size), audio.shape[0]%args.chunk_size] # batches of shape chunk_size, with last batch of size remainder
    batches = [audio[i*el:(i+1)*el] for i, el in enumerate(b_chunks) if el != 0] 
    return batches

def get_logits(model:Wav2Vec2ForCTC, processor:Wav2Vec2Processor, batches:list) -> torch.tensor:
    logits = []
    with torch.no_grad():
        for batch in batches:
            logits.append(model(**{k:v.to(model.device) for k,v in processor(batch, sampling_rate=16000, return_tensors="pt").items()}).logits.squeeze().cpu())
    return torch.vstack(logits)

def apply_softmax(logits:torch.tensor) -> torch.tensor:
    with torch.no_grad():
        smax = torch.nn.LogSoftmax(dim=-1)
        lpz = smax(logits).numpy()
    return lpz

def process(args, audio:np.array, text:str):
    # load model and other stuff
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")
    vocab_dict = processor.tokenizer.get_vocab()
    char_list = [x.lower() for x in vocab_dict.keys()]
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('WARNING: CUDA is not available!') if not torch.cuda.is_available() else None
    model.to(device)
    # get ctc predictions from model
    batches = batch_audio(args, audio)
    lpz = apply_softmax(get_logits(model, processor, batches))
    # segment
    text = text.lower().split()
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / lpz.shape[0] / SAMPLE_RATE # duration in s of each element in output
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, lpz, ground_truth_mat) 
    segments_ = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text) # where the action happens
    segments = get_utterances(gap=0.25, max=7.5, min=2, text=text, segments=segments_)
    return segments

def to_dict(segments:list, fname:str, wav_name:str) -> List[dict]:
    '''
    Converts a list of segment objects to a list of dictionaries
    '''
    return [
        {   
            'parent':wav_name,
            'name': fname.replace(' ', '_')[:-len('.txt')] + '_' + str(i),
            'text': str(segment),
            'start': max(segment.segment_list[0][0] - PADDING, 0), # must be non-negative irrespectively of padding
            'end': segment.segment_list[-1][1] + PADDING,
            'length': len(segment),
            'avg_confidence': segment.avg_confidence(),
            'min_confidence': segment.min_confidence()
        }
        for i, segment in enumerate(segments)
    ]

def save_audio(args, segments_df:pd.DataFrame) -> None:
  '''
  Saves segments from the datafram to individual files
  '''
  print(f'{"-"*50}\n{"Saving audio segments":^50}\n{"-"*50}')
  unique_wavs = segments_df['parent'].unique()
  for wav_name in tqdm(unique_wavs):
    wav, fs = sf.read(os.path.join(args.audio, wav_name))
    for i, segment in segments_df[segments_df['parent']==wav_name].iterrows():
      start = int(segment['start']*fs)
      end = int(segment['end']*fs)
      sf.write(os.path.join(args.output_dir, segment['name'] + '.wav'), wav[start:end], fs)


def main(args) -> None:
    # load csv
    df = pd.read_csv(args.csv)
    # we only need audio that has a corresponding transcription
    df = df.loc[df['Wav_File'] != 'NONE']
    segment_list = []
    # iterate df
    print(f'{"-"*50}\n{"Predicting, and segmenting":^50}\n{"-"*50}')
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # get transcript
        transcript = open_textfile(args, row['Text_File'])
        # get audio
        audio, fs = sf.read(os.path.join(args.audio, row['Wav_File']))
        # segment audio w/ transcript
        segments = process(args, audio.squeeze(), transcript)
        # add to list of dictonary
        segdata = to_dict(segments, row['Text_File'], row['Wav_File'])
        segment_list.extend(segdata)

    print(f'{"-"*50}\n{"Segmented":^50}\n{"-"*50}')
    segments_df = pd.DataFrame(segment_list)
    segments_df.to_csv(args.csv_out, index=False)
    print(f'{"-"*50}\n{"Saved dataframe":^50}\n{"-"*50}')
    #save audio segments
    #save_audio(args, segments_df) #not working...


def check_dirs_and_files_exit(args):
    for item in [args.audio, args.output_dir, args.csv, args.text_files, args.csv]:
        if not os.path.exists(item):
            raise ValueError(f'{item} does not exist!')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment audio files')
    parser.add_argument('--audio', type=str, default='./audio', help='audio directory')
    parser.add_argument('--text_files', type=str, default='./text_files', help='transcript directory')
    parser.add_argument('--csv' , type=str, default='corresponding_audio.csv', help='CSV aligning audio and respective transcripts')
    parser.add_argument('--chunk_size', type=int, default=200000, help='batch size')
    parser.add_argument('--csv_out', type=str, default='segments.csv', help='output csv file')
    parser.add_argument('--output_dir', type=str, default='./segments', help='output directory')
    args = parser.parse_args()
    check_dirs_and_files_exit(args)
    main(args)