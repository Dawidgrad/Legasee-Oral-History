"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
"""
################################################################
# Importing libraries

import sys
import getopt
from utilities import get_transcripts
import subprocess
import sys
import spacy

# Download the en_core_web_sm model
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'h')
opts = dict(opts)

def printHelp():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit()
    
if '-h' in opts:
    printHelp()


################################################################
# Main program function

directory = "../transcripts/ingested"
transcripts = get_transcripts(directory)

# Load the model
nlp = spacy.load('en_core_web_sm')

# Get the NER tags
for batch in transcripts[0]:
    doc = nlp(batch)
    for ent in doc.ents:
        print(ent.text,ent.label_)