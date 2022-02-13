################################################################
# Importing libraries

from flair.data import Sentence
from flair.models import SequenceTagger
from utilities import get_transcripts, write_to_file, TranscriptType

################################################################
# Class definition

class Flair_Entities:
    def get_entities(self):
        directory = "../transcripts/ingested"
        transcript = get_transcripts(TranscriptType.TEST, directory)
        entities = list()

        # Load the NER tagger
        tagger = SequenceTagger.load('ner')
        
        for batch in transcript:
            # Get predictions for the text batch
            sentence = Sentence(batch)
            tagger.predict(sentence)
            entities = entities + sentence.to_dict(tag_type='ner')['entities']
            entities.append('batch_end')

        # Convert format of the Flair entities to universal one
        formatted_entities = self.convert_format(entities)
        
        return formatted_entities

    def convert_format(self, entities):
        formatted_entities = list()

        for entity in entities:
            if entity == 'batch_end':
                formatted_entities.append('batch_end')
                continue
            formatted_entities.append(([entity['start_pos'], entity['end_pos']], entity['labels'][0].value))

        return formatted_entities

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    flair_recogniser = Flair_Entities()
    flair_entities = flair_recogniser.get_entities()

    # Write the result to the output file
    write_to_file("./outputs/flair_results.txt", flair_entities)