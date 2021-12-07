from flair.data import Sentence
from flair.models import SequenceTagger
from utilities import get_transcripts

class Flair_Entities:
    def get_entities(self):
        directory = "../transcripts/ingested"
        transcripts = get_transcripts(directory)
        spans = list()

        for batch in transcripts[0]:
            # make a sentence
            transcript = Sentence(batch)

            # load the NER tagger
            tagger = SequenceTagger.load('ner')

            # run NER over sentence
            tagger.predict(transcript)

            print(transcript)
            print('The following NER tags are found:')

            for entity in transcript.get_spans('ner'):
                print(entity)
                spans.append(entity)

        return spans
