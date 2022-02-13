################################################################
# Importing libraries

from utilities import get_transcript, write_to_file, TranscriptType
from deeppavlov import configs, build_model
import json

################################################################
# Class definition

class DeepPavlov_Entities:
    def get_entities(self):
        directory = "../transcripts/ingested"
        transcript = get_transcript(TranscriptType.TEST, directory)
        entities = list()

        ner_model = build_model(configs.ner.conll2003_m1, download=True)

        for batch in transcript:
            print(batch)
            # Get predictions for the text batch
            entities = ner_model([batch])
            print(entities)

        # Convert format of the DeepPavlov entities to universal one
        # formatted_entities = self.convert_format(entities)
        
        # return formatted_entities

    def convert_format(self, output):
        raw_data = output[0]
        dict_output = json.loads(raw_data)['entities']

        formatted_entities = []

        if 'Date' in dict_output:
            for item in dict_output['Date']:
                formatted_entities.append((item['indices'], 'DATE'))
        
        if 'Location' in dict_output:
            for item in dict_output['Location']:
                formatted_entities.append((item['indices'], 'LOC'))

        if 'Person' in dict_output:
            for item in dict_output['Person']:
                formatted_entities.append((item['indices'], 'PER'))

        if 'Organization' in dict_output:
            for item in dict_output['Organization']:
                formatted_entities.append((item['indices'], 'ORG'))

        return formatted_entities

################################################################
# Main Function

if __name__ == '__main__':
    # Get the Named Entities from GATE API
    pavlov_recogniser = DeepPavlov_Entities()
    pavlov_entities = pavlov_recogniser.get_entities()

    # Write the result to the output file
    write_to_file("./outputs/deeppavlov_results.txt", pavlov_entities)