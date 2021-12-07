from utilities import get_transcript
import subprocess
import sys
import spacy

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