from utilities import get_transcript
from deeppavlov import configs, build_model

class DeepPavlov_Entities:
    def get_entities(self):
        directory = "../transcripts/ingested"
        transcript = get_transcript(directory)
        entities = list()

        ner_model = build_model(configs.ner.ner_ontonotes_bert_torch, download=True)
        ner_model(['Bob Ross lived in Florida'])

        # for batch in transcript:
        #     # Get predictions for the text batch
        #     entities = ner_model([batch])

        # Convert format of the DeepPavlov entities to universal one
        formatted_entities = self.convert_format(entities)
        
        return formatted_entities