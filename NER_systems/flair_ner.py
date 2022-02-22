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

from flair.data import Sentence
from flair.models import SequenceTagger
from utilities import get_transcripts, write_to_file, TranscriptType, tag_transcripts
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

class Flair_Entities:
    def get_entities(self, transcripts):
        # Load the NER tagger
        tagger = SequenceTagger.load('ner')
        entities = list()

        for single_transcript in transcripts:
            for segment in single_transcript:
                sentence = Sentence(segment)
                tagger.predict(sentence)
                entities = entities + sentence.to_dict(tag_type='ner')['entities']
                entities.append('segment_end')
            entities.append('transcript_end')

        # Convert format of the Flair entities to universal one
        formatted_entities = self.convert_format(entities)
        
        return formatted_entities

    def convert_format(self, entities):
        formatted_entities = list()

        for entity in entities:
            if entity == 'segment_end':
                formatted_entities.append('segment_end')
                continue
            if entity == 'transcript_end':
                formatted_entities.append('transcript_end')
                continue
                
            formatted_entities.append(([entity['start_pos'], entity['end_pos']], entity['labels'][0].value))

        return formatted_entities

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    flair_recogniser = Flair_Entities()
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

    flair_entities = flair_recogniser.get_entities(transcripts)

    # Write the result to the output file
    write_to_file("./outputs/flair_results.txt", flair_entities)

    # Use entities to write tagged transcript
    tagged_transcripts = tag_transcripts(flair_entities, transcripts)
    write_to_file("./outputs/flair_tagged_transcript.txt", tagged_transcripts)