import os
import pandas as pd
import json
from enum import Enum

class TranscriptType(Enum):
    ANNOTATION = 1
    OUTPUT = 2
    TEST = 3

# Get the appropriate transcript based on the type of input we want to use
def get_transcript(directory, type):
    result = []

    if type == TranscriptType.ANNOTATION:
        with open('./ner_annotations.jsonl', 'r') as json_file:
            json_list = list(json_file)
        
        for json_str in json_list:
            document = json.loads(json_str)
            result.append(document['data'])

    elif type == TranscriptType.OUTPUT:
        # TODO
        result = []
    elif type == TranscriptType.TEST:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == 'John Roche.tsv':
                    df = pd.read_csv("{}/{}".format(directory, file), sep='\t')
                    cond = (df['Speaker'] != 'New Film') & (df['Speaker'] != 'End of audio')
                    section_list = list(df[cond]['Transcript'])

                    consecutive_n = 5
                    temp = '{} ' * consecutive_n
                    result = [temp.format(*item) for item in zip(*[iter(section_list)] * consecutive_n)] 
                    break

    return result

def write_to_file(directory, data):
    with open(directory, "w") as file:
        for item in data:
            file.write(str(item) + '\n')