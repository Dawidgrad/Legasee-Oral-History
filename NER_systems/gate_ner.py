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
import getopt
import os
import requests
import pandas as pd
from utilities import get_transcripts

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

################################################################
# Main program function

#### Might need to add rate limiting wrapper ####

directory = "../transcripts/ingested"
transcripts = get_transcripts(directory)

gate_output = []
for transcript_part in transcripts[0]:
    gate_output.append(call_gate_api(transcript_part))

print(gate_output)
