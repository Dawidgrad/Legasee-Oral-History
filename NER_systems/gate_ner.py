"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
    -k KEY_ID : API Key ID to your GATE account 
    -p PASSWORD : Password to the GATE API Key
"""

################################################################
# Importing libraries

import sys
import json
import getopt
import requests
from ratelimit import limits, sleep_and_retry
from utilities import get_transcript, write_to_file, TranscriptType

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'hk:p:')
opts = dict(opts)
KEY_ID = opts['-k']
PASSWORD = opts['-p']

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
    @limits(calls=1, period=2)
    def call_gate_api(self, transcript):
        results = []
        endpoint_url = "https://cloud-api.gate.ac.uk/process/annie-named-entity-recognizer?{}={}".format(self.key_id, self.password)
        headers = {
            "Content-Type": "text/plain"
        }

        response = requests.request("POST", endpoint_url, headers = headers, data = transcript.encode('utf-8'))
        results.append(response.text)

        return results

    def get_entities(self):
        directory = "../transcripts/ingested"
        transcript = get_transcript(TranscriptType.TEST, directory)

        entities = []
        for batch in transcript:
            gate_output = self.call_gate_api(batch)
            print(gate_output)
            entities = entities + self.convert_format(gate_output)
            entities.append('batch_end')
            break

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
    gate_entities = gate_recogniser.get_entities()

    # Write the result to the output file
    write_to_file("./outputs/gate_results.txt", gate_entities)