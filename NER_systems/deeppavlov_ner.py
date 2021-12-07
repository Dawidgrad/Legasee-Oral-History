from utilities import get_transcripts

class DeepPavlov_Entities:
    def get_entities(self):
        directory = "../transcripts/ingested"
        transcripts = get_transcripts(directory)

        # Do the rest on Linux (not working on Windows)
