################################################################
# Importing libraries

from utilities import get_transcript, write_to_file
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

################################################################
# Class definition

class Stanford_Entities:
    def __init__(self):
        nltk.download('punkt')

    def get_entities(self):
        directory = "../transcripts/ingested"
        transcript = get_transcript(directory)

        # Load the model
        st = StanfordNERTagger('stanford_models/english.all.3class.distsim.crf.ser.gz',
                            'stanford_models/stanford-ner.jar',
                            encoding='utf-8')

        entities = list()

        # Get the NER tags
        for batch in transcript:
            tokenized_text = word_tokenize(batch)
            classified_text = st.tag(tokenized_text)

            # Convert to universal format
            formatted_entities = self.convert_format(classified_text, batch)
            entities = entities + formatted_entities
            entities.append('batch_end')

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
    write_to_file("./stanford_results.txt", stanford_entities)