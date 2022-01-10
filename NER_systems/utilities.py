import os
import pandas as pd

def get_transcript(directory):    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'John Roche.tsv':
                df = pd.read_csv("{}/{}".format(directory, file), sep='\t')
                cond = (df['Speaker'] != 'New Film') & (df['Speaker'] != 'End of audio')
                section_list = list(df[cond]['Transcript'])

                consecutive_n = 5
                temp = '{} ' * consecutive_n
                full_transcript = [temp.format(*item) for item in zip(*[iter(section_list)] * consecutive_n)] 
                break

    return full_transcript

def write_to_file(directory, data):
    with open(directory, "w") as file:
        for item in data:
            file.write(str(item) + '\n')
