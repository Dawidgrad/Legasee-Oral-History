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

from utilities import get_transcripts, write_to_file, TranscriptType, tag_transcripts
import subprocess
import sys
import spacy
import getopt
import os

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'hd:ao')
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

class Spacy_Entities:
    def __init__(self):
        # Download the en_core_web_sm model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    def get_entities(self, transcripts): 
        # Load the model
        nlp = spacy.load('en_core_web_sm')
        entities = list()
        ignored_labels = ['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

        # Get the NER tags
        for single_transcript in transcripts:
            for segment in single_transcript:
                doc = nlp(segment)
                for ent in doc.ents:
                    if ent.label_ not in ignored_labels:
                        entities = entities + [([ent.start_char, ent.end_char], ent.label_)]
                entities.append('segment_end')
            entities.append('transcript_end')
        
        return entities

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    spacy_recogniser = Spacy_Entities()
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

    spacy_entities = spacy_recogniser.get_entities(transcripts)

    # Write the result to the output file
    write_to_file(f'{BASE_DIR}/ner_output/spacy_index_results.txt', spacy_entities)

    # Use entities to write tagged transcript
    tagged_transcripts = tag_transcripts(spacy_entities, transcripts)
    write_to_file(f'{BASE_DIR}/ner_output/spacy_tagged_transcript.txt', tagged_transcripts)
