"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit

    Specify ONE method of transcript type handling:
    -a ANNOTATION : uses annotation transcripts (dictionary format)
    -o ASR_OUTPUT : uses ASR system output (WIP)
    -t TEST : uses a test file (John Roche from batch 0)
"""

################################################################
# Importing libraries

from utilities import get_transcripts, write_to_file, TranscriptType
import subprocess
import sys
import spacy
import getopt

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'haot')
opts = dict(opts)

def printHelp():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit()
    
if '-h' in opts:
    printHelp()

if ('-a' not in opts) and ('-o' not in opts) and ('-t' not in opts):
    print("\n** ERROR: must specify transcription handling method **", file=sys.stderr)
    printHelp()

options_count = 0
options_count += 1 if '-a' in opts else 0
options_count += 1 if '-o' in opts else 0
options_count += 1 if '-t' in opts else 0

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

    def get_entities(self): 
        # Load the model
        nlp = spacy.load('en_core_web_sm')
        entities = list()
        ignored_labels = ['LANGUAGE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
        transcripts = []

        # Decide on the transcription type
        if '-a' in opts:
            dictionaries = get_transcripts(TranscriptType.ANNOTATION, '')

            for dictionary in dictionaries:
                single_transcript = []
                for key, value in dictionary.items():
                    single_transcript = [*single_transcript, *value]
                transcripts.append(single_transcript) 
                
        elif '-o' in opts:
            #TODO
            print('WIP')
        else:
            directory = "../transcripts/ingested"
            transcripts = get_transcripts(TranscriptType.TEST, directory)

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
    spacy_entities = spacy_recogniser.get_entities()

    # Write the result to the output file
    write_to_file("./outputs/spacy_results.txt", spacy_entities)