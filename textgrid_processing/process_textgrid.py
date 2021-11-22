from os import listdir
grid_ext = '.TextGrid'
wav_ext = '.wav'
import pandas as pd
import soundfile as sf
import numpy as np

def chain_textgrids(txt_grids:list) -> str:
    return " ".join([read_textgrid(el) for el in txt_grids])

def get_textgrid(wav_file:str, wav_path=None, textgrid_path=None) -> list:
    dir = listdir(textgrid_path)
    all_textgrids = [el for el in dir if el[-len(grid_ext):] == grid_ext]
    fname = wav_file[:-len(wav_ext)]
    textgrids = [el for el in all_textgrids if el[:len(fname)] == fname]
    return sorted(textgrids)

def read_textgrid(fn:str) -> str:
    with open(fn, 'r') as f:
        lines = f.read().split('\n')
    intervals = [i for i, line in enumerate(lines) if line == '"IntervalTier"']
    word = [i for i, line in enumerate(lines) if line == '"word"' and lines[i-1] == '"IntervalTier"'][0]
    target_ids = [[intervals[i]+2,intervals[i+1]-1] for i, el in enumerate(intervals) if el == word-1][0]
    transcript = " ".join([el.split('"')[1] for el in lines[target_ids[0]:target_ids[1]] if '"' in el and el != '"sp"']).strip()
    return transcript

def save_output(text:str, outfile:str):
    with open(outfile, 'w') as f:
        f.write(text)
        
# utterance extraction

class Utterances():
    def __init__(self, wav_file:str, padding=0.2, gap=0.4, out_path='out/', wav_path='', textgrid_path=None):
        self.wav_file = wav_file
        self.out_path = out_path
        self.name = self.wav_file[:-len(wav_ext)]
        self.wav_path = wav_path
        self.textgrid_path = textgrid_path
        self.textgrid_files = get_textgrid(wav_file, wav_path, textgrid_path)
        self.textgrids_ = [self.read_wordIntervals(el) for el in self.textgrid_files]
        self.text_sequence = self.chain_sequences(self.textgrids_)
        self.wav, self.sf = sf.read(self.wav_path + self.wav_file)
        self.padding = padding
        self.gap = gap
        self.process_sequence()


    def read_wordIntervals(self, fn:str) -> list:
        with open(fn, 'r') as f:
            lines = f.read().split('\n')
        intervals = [i for i, line in enumerate(lines) if line == '"IntervalTier"']
        word = [i for i, line in enumerate(lines) if line == '"word"' and lines[i-1] == '"IntervalTier"'][0]
        target_ids = [[intervals[i]+2,intervals[i+1]-1] for i, el in enumerate(intervals) if el == word-1][0]
        word_is = [el for el in lines[target_ids[0]:target_ids[1]] if 1==1 and el != '"sp"']
        return word_is

    def chain_sequences(self, wordIntervals:list) -> list:
        return [item for sublist in wordIntervals for item in sublist] #expand lists

    def process_sequence(self) -> list:
        word_data = []
        for i, el in enumerate(self.text_sequence):
            if i == 0: # skip the first element
                continue
            if '"' in el and el != '"sp"': # if it's a word
                word = el.split('"')[1]
                word_start = self.text_sequence[i-1]
                if i == len(self.text_sequence)-1: # if it's the last word
                    word_end = word_start + 0.25 # if end time is not specified, assume length of 0.25 seconds
                else:
                    word_end = self.text_sequence[i+1]
                word_data.append({
                    'word': word,
                    'start': float(word_start),
                    'end': float(word_end)
                })
        ################################################################################
        utterances = []
        padding = self.padding # start and end padding
        gap = self.gap# gap between words for an utterance
        for i, el in enumerate(word_data):
            if i == 0:  # if first element then start a new utturance
                utterances.append({
                    'word_lst': [el['word']],
                    'start': el['start'],
                    'end': el['end']
                })
            else:
                if (word_data[i]['start'] - word_data[i-1]['end']) < gap: # if gap between words is less than gap, then add word to utterance
                    utterances[-1]['word_lst'].append(el['word'])
                    utterances[-1]['end'] = el['end']
                else: # if gap between words is greater than gap, then start a new utterance
                    utterances[-1]['gap'] = word_data[i]['start'] - word_data[i-1]['end']
                    utterances.append({
                        'word_lst': [el['word']],
                        'start': el['start'],
                        'end': el['end']
                    })
        for i, utt in enumerate(utterances):
            utt['start'] = utt['start'] - padding
            utt['end'] = utt['end'] + padding
            utt['length'] = utt['end'] - utt['start']
            utt['sentence'] = " ".join(utt['word_lst'])

        self.sentences = [{'text':utt['sentence'], 'start':utt['start'], 'end':utt['end'], 'length':utt['length']} for utt in utterances if utt['length'] >= 2 and utt['length'] <= 10]

    def get_utterance(self, start, end):
        return self.wav[int(start*self.sf):int(end*self.sf)]

    def save_output(self, sequence:np.array, n:int):
        f_out = f'{self.name}_u-{n}.wav'
        sf.write(self.out_path+f_out, sequence, self.sf)
        return f_out

    def save_utterances(self) -> pd.DataFrame:
        data = {
            'parent_wav':[],
            'utterance_wav':[],
            'label':[],
            'start':[],
            'end':[],
            'length':[]
        }
        for i, utt in enumerate(self.sentences):
            sequence = self.get_utterance(utt['start'], utt['end'])
            out_name = self.save_output(sequence, n=i)
            data['parent_wav'].append(self.wav_file)
            data['utterance_wav'].append(out_name)
            data['label'].append(utt['text'])
            data['start'].append(utt['start'])
            data['end'].append(utt['end'])
            data['length'].append(utt['length'])

        return pd.DataFrame(data)



    
