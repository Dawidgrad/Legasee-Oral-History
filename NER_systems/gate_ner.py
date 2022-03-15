"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
    -d : directory of the NER package
    -k KEY_ID : API Key ID to your GATE account 
    -p PASSWORD : Password to the GATE API Key

    Specify ONE method of transcript type handling:
    -a ANNOTATION : uses annotation transcripts (dictionary format)
    -o ASR_OUTPUT : uses ASR system output
"""

################################################################
# Importing libraries

import os
import sys
import json
import getopt
import requests
from ratelimit import limits, sleep_and_retry
from utilities import get_transcripts, write_to_file, TranscriptType, tag_transcripts

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'hd:aok:p:')
opts = dict(opts)
KEY_ID = opts['-k']
PASSWORD = opts['-p']
BASE_DIR = opts['-d']

def printHelp():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit()
    
if '-h' in opts:
    printHelp()

if '-k' not in opts:
    print("\n** ERROR: must specify API Key ID (opt: -k KEY_ID) **", file=sys.stderr)
    printHelp()

if '-p' not in opts:
    print("\n** ERROR: must specify API Key Password (opt: -p PASSWORD) **", file=sys.stderr)
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

class Gate_Entities:
    def __init__(self, key_id, password) -> None:
        self.key_id = key_id
        self.password = password
        
    # Rate limit API calls to GATE
    @sleep_and_retry
    @limits(calls=1, period=1)
    def call_gate_api(self, transcript):
        results = []
        endpoint_url = 'https://cloud-api.gate.ac.uk/process/annie-named-entity-recognizer'
        headers = {
            "Content-Type": "text/plain"
        }

        response = requests.post(endpoint_url,
                                auth=(self.key_id, self.password),
                                data = transcript.encode('utf-8'),
                                headers = headers)
        results.append(response.text)

        return results

    def get_entities(self, transcripts):
        entities = list()
       
        # Get the NER tags
        for single_transcript in transcripts:
            for segment in single_transcript:
                gate_output = self.call_gate_api(segment)
                print(gate_output)
                entities = entities + self.convert_format(gate_output)
                entities.append('segment_end')
            entities.append('transcript_end')
        
        return entities

    def convert_format(self, output):
        raw_data = output[0]
        dict_output = json.loads(raw_data)['entities']

        formatted_entities = []

        if 'Date' in dict_output:
            for item in dict_output['Date']:
                formatted_entities.append((item['indices'], 'DATE'))
        
        if 'Location' in dict_output:
            for item in dict_output['Location']:
                formatted_entities.append((item['indices'], 'LOC'))

        if 'Person' in dict_output:
            for item in dict_output['Person']:
                formatted_entities.append((item['indices'], 'PER'))

        if 'Organization' in dict_output:
            for item in dict_output['Organization']:
                formatted_entities.append((item['indices'], 'ORG'))

        return formatted_entities

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    gate_recogniser = Gate_Entities(KEY_ID, PASSWORD)
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

    gate_entities = gate_recogniser.get_entities(transcripts)

    # Write the result to the output file
    write_to_file(f'{BASE_DIR}/ner_output/gate_results.txt', gate_entities)

    # Use entities to write tagged transcript
    tagged_transcripts = tag_transcripts(gate_entities, transcripts)
    write_to_file(f'{BASE_DIR}/ner_output/gate_tagged_transcript.txt', tagged_transcripts)