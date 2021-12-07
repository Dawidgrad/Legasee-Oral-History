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

from utilities import get_transcript
from flair_ner import Flair_Entities
# from deeppavlov_ner import DeepPavlov_Entities
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

# flair_recogniser = Flair_Entities()
# flair_entities = flair_recogniser.get_entities()
# print(flair_entities)

# pavlov_recogniser = DeepPavlov_Entities()
# pavlov_entities = pavlov_recogniser.get_entities()
# print(pavlov_entities)

# spacy_recogniser = Spacy_Entities()
# spacy_entities = spacy_recogniser.get_entities()
# print(spacy_entities)

# stanford_recogniser = Stanford_Entities()
# stanford_entities = stanford_recogniser.get_entities()
# print(stanford_entities)

gate_recogniser = Gate_Entities(KEY_ID, PASSWORD)
gate_entities = gate_recogniser.get_entities()
print(gate_entities)