import requests
from utilities import get_transcripts

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
        transcripts = get_transcripts(directory)

        gate_output = []
        for transcript_part in transcripts[0]:
            gate_output.append(call_gate_api(transcript_part))

        print(gate_output)

        return gate_output
