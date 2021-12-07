from utilities import get_transcripts
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

class Stanford_Entities:
    def __init__(self) -> None:
        nltk.download('punkt')

    def get_entities(self):
        directory = "../transcripts/ingested"
        transcripts = get_transcripts(directory)

        # Load the model
        st = StanfordNERTagger('stanford_models/english.all.3class.distsim.crf.ser.gz',
                            'stanford_models/stanford-ner.jar',
                            encoding='utf-8')

        entities = list()

        # Get the NER tags
        for batch in transcripts[0]:
            tokenized_text = word_tokenize(batch)
            classified_text = st.tag(tokenized_text)

            print(classified_text)
            entities.append(classified_text)

        return entities
