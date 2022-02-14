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
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import getopt
import sys

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

class Stanford_Entities:
    def __init__(self):
        nltk.download('punkt')

    def get_entities(self):
        # Load the model
        st = StanfordNERTagger('stanford_models/english.all.3class.distsim.crf.ser.gz',
                            'stanford_models/stanford-ner.jar',
                            encoding='utf-8')
        entities = list()
        transcripts = []

        if '-a' in opts:
            dictionaries = get_transcripts(TranscriptType.ANNOTATION, '')
            transcripts = []

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
                tokenized_text = word_tokenize(segment)
                classified_text = st.tag(tokenized_text)

                # Convert to universal format
                formatted_entities = self.convert_format(classified_text, segment)
                entities = entities + formatted_entities
                entities.append('segment_end')
            entities.append('transcript_end')
        
        return entities

    def convert_format(self, entities, transcript):
            formatted_entities = list()
            offset = 0

            for entity in entities:
                if entity[1] != 'O':
                    span_start = transcript.find(entity[0], offset)
                    span_end = span_start + len(entity[0])
                    ner_class = entity[1][0:3]
                    formatted_entities.append(([span_start, span_end], ner_class))
                    offset = span_end

            return formatted_entities

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    stanford_recogniser = Stanford_Entities()
    stanford_entities = stanford_recogniser.get_entities()

    # Write the result to the output file
    write_to_file("./outputs/stanford_results.txt", stanford_entities)