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
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

nltk.download('punkt')

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
st = StanfordNERTagger('stanford_models/english.all.3class.distsim.crf.ser.gz',
					   'stanford_models/stanford-ner.jar',
					   encoding='utf-8')

# Get the NER tags
for batch in transcripts[0]:
    tokenized_text = word_tokenize(batch)
    classified_text = st.tag(tokenized_text)

    print(classified_text)
