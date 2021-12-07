from utilities import get_transcripts
import subprocess
import sys
import spacy

class Spacy_Entities:
    def __init__(self):
        # Download the en_core_web_sm model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    def get_entities(self):  
        directory = "../transcripts/ingested"
        transcripts = get_transcripts(directory)

        # Load the model
        nlp = spacy.load('en_core_web_sm')
        entities = list()

        # Get the NER tags
        for batch in transcripts[0]:
            doc = nlp(batch)
            for ent in doc.ents:
                print(ent.text,ent.label_)
                entities.append((ent.text, ent.label_))
        
        return entities
                