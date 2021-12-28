from enum import unique
import soundfile as sf
from os.path import join
import pandas as pd
from tqdm import tqdm

TARGET_DIR = "segments" 
DATAFRAME = "segments.csv"
AUDIO = "audio"

def main():
    df = pd.read_csv(DATAFRAME)
    unique_wavs = df['parent'].unique()
    for wav_name in tqdm(unique_wavs):
        wav, fs = sf.read(join(AUDIO, wav_name))
        print(wav_name)
        for i, segment in df[df['parent']==wav_name].iterrows():
            print(segment)
            start = int(segment['start']*fs)
            end = int(segment['end']*fs)
            fname = join(TARGET_DIR, segment['name'] + '.wav')
            print(fname)
            sf.write(fname, wav[start:end], fs)

if __name__ == "__main__":
    print(f'{"-"*50}\n{"Extracting and saving segments!!":^50}\n{"-"*50}')
    main()