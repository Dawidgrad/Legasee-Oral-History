from re import S
from typing import Dict, List
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np
import soundfile as sf

FS = 16000 # ASR model only handles 16kHz audio so this is hardcoded

'''
TODO: make sure value dict maintains order! this might happen automatically in python 3.X
'''

def batch_wav(wav, batch_len) -> List[np.ndarray]:
  evensteps = wav.shape[0]//(batch_len*FS)
  batches = [wav[int(i*batch_len*FS):int((i+1)*batch_len*FS)] for i in range(evensteps)]
  if evensteps != int(wav.shape[0])/int(batch_len*FS):
    batches.append(wav[int(batch_len*FS*evensteps):])
  return batches

def proc_batches(encoder, wav, rate, batch_len) -> np.ndarray:
  batches = batch_wav(wav,batch_len)
  embeds = []
  wsplits = []
  for batch in batches:
    _, cont_embeds, wav_splits = encoder.embed_utterance(batch, return_partials=True, rate=rate)
    embeds.append(cont_embeds)
    wsplits.append(wav_splits)
  return np.concatenate(embeds)

def get_speaker_list(similarity_dict, sr):
    speaker_list = []
    for i in range(list(similarity_dict.values())[0].shape[0]):
        sp1 = {
            'conf':similarity_dict['Interviewer'][i],
            'speaker':'Interviewer',
            'start':i*sr+sr,
            'end':(i+1)*sr+sr/2
        }
        sp2 = {
            'conf':similarity_dict['Interviewee'][i],
            'speaker':'Interviewee',
            'start':i*sr+sr,
            'end':(i+1)*sr+sr/2
        }
        mx = max(sp1, sp2, key=lambda x:x['conf'])
        if len(speaker_list) == 0:
            speaker_list.append(mx)
        elif speaker_list[-1]['speaker'] == mx['speaker']:
            speaker_list[-1]['end'] = mx['end']+sr/2
            speaker_list[-1]['conf'] = mx['conf'] # eh fix pls
        else:
            speaker_list.append(mx)
    return speaker_list

def diarize(wav, martin_s, rate=1.3, blen=1000) -> List[Dict]:
    encoder = VoiceEncoder("cpu")
    cont_embeds = proc_batches(encoder, wav, rate, blen)
    martin_embed, all_embed = [encoder.embed_utterance(speaker_wav) for speaker_wav in [martin_s, wav]]
    # @ = matrix multiplication
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in zip(['Interviewer', 'Interviewee'], [martin_embed, all_embed])} 
    speaker_list = get_speaker_list(similarity_dict, 1/rate)
    return speaker_list


def run_diarization(martin_s, file=None, wav=None):
    if file is not None:
        audio, fs = sf.read(file)
    else:
        audio, fs = wav, 16000
    # Load the speaker vector model and the webrtc VAD model
    speaker_list = diarize(audio, martin_s)
    # remove conf key from dicts
    speaker_list = [{k:v for k,v in sp.items() if k != 'conf'} for sp in speaker_list] # remove conf key until it's fixed in the code above
    return speaker_list

