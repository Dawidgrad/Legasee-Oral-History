################################################################
# Importing libraries

from flair.data import Sentence
from flair.models import SequenceTagger
from utilities import get_transcript, write_to_file

################################################################
# Class definition

class Flair_Entities:
    def get_entities(self):
        directory = "../transcripts/ingested"
        transcript = get_transcript(directory)
        entities = list()

        # Load the NER tagger
        tagger = SequenceTagger.load('ner')
        
        for batch in transcript:
            # Get predictions for the text batch
            sentence = Sentence(batch)
            tagger.predict(sentence)
            entities = entities + sentence.to_dict(tag_type='ner')['entities']

        # Convert format of the Flair entities to universal one
        formatted_entities = self.convert_format(entities)
        
        return formatted_entities

    def convert_format(self, entities):
        formatted_entities = list()

        for entity in entities:
            formatted_entities.append(([entity['start_pos'], entity['end_pos']], entity['labels'][0].value))

        return formatted_entities

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    flair_recogniser = Flair_Entities()
    flair_entities = flair_recogniser.get_entities()

    # Write the result to the output file
    write_to_file("./flair_results.txt", flair_entities)