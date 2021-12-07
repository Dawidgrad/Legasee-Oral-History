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
import getopt
from utilities import get_transcripts
from flair_ner import Flair_Entities
from deeppavlov_ner import DeepPavlov_Entities
from spacy_ner import Spacy_Entities
from stanford_ner import Stanford_Entities
from gate_ner import Gate_Entities

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
# Main program function

directory = "../transcripts/ingested"
transcripts = get_transcripts(directory)

# print(transcripts[1])