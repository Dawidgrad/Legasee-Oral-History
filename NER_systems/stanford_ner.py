from utilities import get_transcript
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

class Stanford_Entities:
    def __init__(self) -> None:
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

            formatted_entities = self.convert_format(classified_text, batch)
            entities = entities + formatted_entities

            break # Single batch for now


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