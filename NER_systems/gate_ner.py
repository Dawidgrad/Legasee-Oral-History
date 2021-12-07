import requests
from utilities import get_transcript
import json

class Gate_Entities:
    def __init__(self, key_id, password) -> None:
        self.key_id = key_id
        self.password = password
        
    def call_gate_api(self, transcript):
        results = []
        endpoint_url = "https://cloud-api.gate.ac.uk/process/annie-named-entity-recognizer?{}={}".format(self.key_id, self.password)
        headers = {
            "Content-Type": "text/plain"
        }

        response = requests.request("POST", endpoint_url, headers = headers, data = transcript.encode('utf-8'))
        results.append(response.text)

        return results

    def get_entities(self):
        #### Might need to add rate limiting wrapper ####
        directory = "../transcripts/ingested"
        transcript = get_transcript(directory)

        entities = []
        for batch in transcript:
            gate_output = self.call_gate_api(batch)
            entities = entities + self.convert_format(gate_output)
            break

        return entities

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

        if 'Organisation' in dict_output:
            for item in dict_output['Organisation']:
                formatted_entities.append((item['indices'], 'ORG'))

        return formatted_entities