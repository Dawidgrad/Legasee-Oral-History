################################################################
# Importing libraries

from utilities import get_transcript, write_to_file
import subprocess
import sys
import spacy

################################################################
# Class definition

class Spacy_Entities:
    def __init__(self):
        # Download the en_core_web_sm model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    def get_entities(self): 
        directory = "../transcripts/ingested"
        transcript = get_transcript(directory)

        # Load the model
        nlp = spacy.load('en_core_web_sm')
        entities = list()
        ignored_labels = ['LANGUAGE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

        # Get the NER tags
        for batch in transcript:
            doc = nlp(batch)
            for ent in doc.ents:
                if ent.label_ not in ignored_labels:
                    entities = entities + [([ent.start_char, ent.end_char], ent.label_)]
        
        return entities

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    spacy_recogniser = Spacy_Entities()
    spacy_entities = spacy_recogniser.get_entities()

    # Write the result to the output file
    write_to_file("./spacy_results.txt", spacy_entities)