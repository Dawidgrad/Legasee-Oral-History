from flair.data import Sentence
from flair.models import SequenceTagger
from utilities import get_transcript

class Flair_Entities:
    def get_entities(self):
        directory = "../transcripts/ingested"
        transcript = get_transcript(directory)
        entities = list()

        # Load the NER tagger
        tagger = SequenceTagger.load('ner')
        
        for batch in transcript:
            # Create a sentence
            sentence = Sentence(batch)

            # Run NER over sentence
            tagger.predict(sentence)

            entities = entities + sentence.to_dict(tag_type='ner')['entities']

            break # single iteration for testing

        formatted_entities = self.convert_format(entities)
        
        return formatted_entities

    def convert_format(self, entities):
        formatted_entities = list()

        for entity in entities:
            formatted_entities.append(([entity['start_pos'], entity['end_pos']], entity['labels'][0].value))

        return formatted_entities
