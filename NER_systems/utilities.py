import os
import pandas as pd
import json
import re
from enum import Enum

class TranscriptType(Enum):
    ANNOTATION = 1
    OUTPUT = 2
    TEST = 3

# Get the appropriate transcript based on the type of input we want to use
def get_transcript(type, directory = ''):
    result = []
    
    if type == TranscriptType.ANNOTATION:
        # Get the json file
        with open('./ner_annotations.jsonl', 'r') as json_file:
            json_list = list(json_file)
        
        raw_transcripts = []
        # Extract the raw transcripts from json file
        for json_str in json_list:
            document = json.loads(json_str)
            raw_transcripts.append(document['data'])
        
        result = segment_trascripts(raw_transcripts)
        
        # with open('segmenting.txt', "w") as file:
        #     for dictionary in result:
        #         for key, value in dictionary.items():
        #             file.write(str(key) + ': ' + str(value) + '\n')
    
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

def segment_trascripts(raw_transcripts):
    result = []
    timestamp_regex = r'(\d{2}:\d{2}:\d{2})'

    # Split transcripts into segments
    for transcript in raw_transcripts:
        split_transcript = re.split(timestamp_regex, transcript)
        split_transcript = split_transcript[3:]

        # Create a dictionary of timestamp keys and transcript text value
        transcript_dict = dict()

        for i in range(0, len(split_transcript), 2):
            transcript_dict[split_transcript[i]] = split_transcript[i + 1]

        # Split the transcript into similar segment sizes
        segment_size = 1000
        segmented_dict = dict()
        for key, value in transcript_dict.items():
            segmented_dict[key] = []
            idx = 0
            while True:
                # Find how far is the closest full stop (relative to segment size)
                idx_limit = idx + segment_size if (idx + segment_size) < len(value) else (len(value) - 1)
                full_stop_idx = value[:idx_limit].rfind('.') # not guaranteed that it's not out of index + start from idx possibly
                triple_dot_idx = value[:idx_limit].rfind('…')
                if (full_stop_idx == -1):
                    full_stop_idx = len(value)

                end_idx = full_stop_idx if full_stop_idx > triple_dot_idx else triple_dot_idx

                # Get the segment from current index to closest full stop
                segment = value[idx:end_idx + 1]

                # Preprocess the segment and add it to the output dictionary
                segment = preprocess_segment(segment)

                if (len(segment) > 5):
                    segmented_dict[key].append(segment)
                idx = end_idx + 1
                
                # Check if the last segment has been encountered already
                if (idx > (len(value) - segment_size)):
                    last_segment = value[end_idx:]
                    last_segment = preprocess_segment(last_segment)
                    if (len(last_segment) > 5):
                        segmented_dict[key].append(last_segment)
                    break

        result.append(segmented_dict)

    return result

def preprocess_segment(segment):
    # Remove newlines
    preprocessed_segment = segment.replace('\n', ' ')

    # Remove "Start of Film" and "End of Film" text
    pattern_start = r'\*\* Start of Film [0-9]*'
    pattern_end = r'End of Films'
    preprocessed_segment = re.sub(pattern_start, '', preprocessed_segment)
    preprocessed_segment = re.sub(pattern_end, '', preprocessed_segment)

    # Remove whitespaces
    preprocessed_segment = preprocessed_segment.strip()

    return preprocessed_segment

def write_to_file(directory, data):
    with open(directory, "w") as file:
        for item in data:
            file.write(str(item) + '\n')