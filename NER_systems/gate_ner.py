"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
    -k KEY_ID : API Key ID to your GATE account 
    -p PASSWORD : Password to the API Key
"""
################################################################
# Importing libraries

import sys
import re
import getopt
import os
import requests
import pandas as pd

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
# Defining functions

def call_gate_api(transcript):
    results = []
    endpoint_url = "https://cloud-api.gate.ac.uk/process/annie-named-entity-recognizer?{}={}".format(KEY_ID, PASSWORD)
    headers = {
        "Content-Type": "text/plain"
    }

    response = requests.request("POST", endpoint_url, headers = headers, data = transcript.encode('utf-8'))
    results.append(response.text)

    return results

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


################################################################
# Main program function

directory = "../transcripts/ingested"
transcripts = get_transcripts(directory)

#### Might need to add rate limiting wrapper ####

# for transcript in transcripts:
#     print(transcript)
#     gate_output = call_gate_api(transcript)
# gate_output = call_gate_api(transcripts[0])

# print(gate_output)

gate_output = []
for transcript_part in transcripts[0]:
    gate_output.append(call_gate_api(transcript_part))

print(gate_output)
