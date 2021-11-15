"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
"""
################################################################
# Importing libraries

import sys
import getopt
from flair.data import Sentence
from flair.models import SequenceTagger
from utilities import get_transcripts

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
# Defining functions

# def get_transcripts(directory):    
#     transcripts = []

#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.tsv'):
#                 df = pd.read_csv("{}/{}".format(directory, file), sep='\t')
#                 cond = (df['Speaker'] == 'Interviewer') | (df['Speaker'] == 'Interviewee')
#                 full_transcript = " ".join(list(df[cond]['Transcript']))
#                 transcripts.append(full_transcript)

#     return transcripts


################################################################
# Main program function

directory = "../transcripts/ingested"
transcripts = get_transcripts(directory)

for batch in transcripts[0]:
    # make a sentence
    transcript = Sentence(batch)

    # load the NER tagger
    tagger = SequenceTagger.load('ner')

    # run NER over sentence
    tagger.predict(transcript)

    print(transcript)
    print('The following NER tags are found:')

    for entity in transcript.get_spans('ner'):
        print(entity)
