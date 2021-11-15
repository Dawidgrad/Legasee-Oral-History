import os
import pandas as pd

def get_transcripts(directory):    
    transcripts = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tsv'):
                df = pd.read_csv("{}/{}".format(directory, file), sep='\t')
                cond = (df['Speaker'] == 'Interviewer') | (df['Speaker'] == 'Interviewee')
                section_list = list(df[cond]['Transcript'])

                consecutive_n = 5
                temp = '{} ' * consecutive_n
                full_transcript = [temp.format(*item) for item in zip(*[iter(section_list)] * consecutive_n)] 
                transcripts.append(full_transcript)

    return transcripts