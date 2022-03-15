"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
    -d : directory of the NER package

    Specify ONE method of transcript type handling:
    -a ANNOTATION : uses annotation transcripts (dictionary format)
    -o ASR_OUTPUT : uses ASR system output
"""

################################################################
# Importing libraries

import os
from utilities import get_transcripts, write_to_file, TranscriptType
from deeppavlov import configs, build_model
import getopt
import sys
from tqdm import tqdm

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'hao')
opts = dict(opts)
BASE_DIR = opts['-d']

def printHelp():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit()
    
if '-h' in opts:
    printHelp()

if ('-a' not in opts) and ('-o' not in opts):
    print("\n** ERROR: must specify transcription handling method **", file=sys.stderr)
    printHelp()

if '-d' not in opts:
    print("\n** ERROR: must specify directory of the NER package **", file=sys.stderr)
    printHelp()

options_count = 0
options_count += 1 if '-a' in opts else 0
options_count += 1 if '-o' in opts else 0

if options_count > 1:
    print("\n** ERROR: cannot use more than one transcription handling method **", file=sys.stderr)
    printHelp()

if len(args) > 0:
    print("\n** ERROR: no arg files - only options! **", file=sys.stderr)
    printHelp()

################################################################
# Class definition

class DeepPavlov_Entities:
    def get_entities(self):
        # Load the model
        ner_model = build_model(configs.ner.ner_ontonotes_bert_torch, download=True)
        entities = list()
        transcripts = []
        
        # Decide on the transcription type
        if '-a' in opts:
            dictionaries = get_transcripts(TranscriptType.ANNOTATION, f'{BASE_DIR}/ner_annotations.jsonl')

            for dictionary in dictionaries:
                single_transcript = []
                for key, value in dictionary.items():
                    single_transcript = [*single_transcript, *value]
                transcripts.append(single_transcript) 
                
        elif '-o' in opts:
            directory = f'{BASE_DIR}/input'
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    transcript = get_transcripts(TranscriptType.OUTPUT, f'{directory}/{filename}')
                    transcripts.append(transcript)

        entities = list()
        # Get the NER tags
        for single_transcript in transcripts:
            for segment in tqdm(single_transcript):
                entities.append(ner_model([segment]))
                entities.append('segment_end')
            entities.append('transcript_end')

        return entities        

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    pavlov_recogniser = DeepPavlov_Entities()
    pavlov_entities = pavlov_recogniser.get_entities()

    # Write the result to the output file
    write_to_file(f'{BASE_DIR}/ner_output/deeppavlov_results.txt', pavlov_entities)

    tagged_transcripts = list()

    is_entity = False
    for entity in pavlov_entities:
        if entity == 'segment_end' or entity == 'transcript_end':
            continue

        segment_text = ''
        for token, tag in zip(entity[0][0], entity[1][0]):
            if tag[0] == 'B':
                label = tag[2:]
                segment_text = segment_text + f' <{label}>' + token
                is_entity = True
            elif tag[0] == 'I':
                segment_text = segment_text + ' ' + token
            elif tag[0] == 'O':
                if is_entity:
                    segment_text = segment_text + f'<\{label}> ' + token
                    is_entity = False
                else:
                    if segment_text == '':
                        segment_text = token
                    else:
                        segment_text = segment_text + ' ' + token

        tagged_transcripts.append(segment_text)
        

    write_to_file(f'{BASE_DIR}/ner_output/deeppavlov_tagged_transcript.txt', tagged_transcripts)
              
          