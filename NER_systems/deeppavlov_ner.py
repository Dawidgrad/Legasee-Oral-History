from utilities import get_transcript

class DeepPavlov_Entities:
    def get_entities(self):
        directory = "../transcripts/ingested"
        transcript = get_transcript(directory)

        # Do the rest on Linux (not working on Windows)
