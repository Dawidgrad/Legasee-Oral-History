import docx
from docx.enum.text import WD_COLOR_INDEX
from utilities import get_transcript

def read_results(path):
    results = list()
    entities = list()

    with open(path) as file:
        line = file.readline()
        while line != '':
            if line == 'batch_end\n':
                results.append(entities)
                line = file.readline()
                entities = list()
                continue
            entities.append(eval(line))
            line = file.readline()

    return results

def write_to_doc(transcript, results):
    for batch, entities in zip(transcript, results):
        para = doc.add_paragraph()
        prev_idx = 0

        for entity in entities:
            start_idx = entity[0][0]
            end_idx = entity[0][1]

            para.add_run(batch[prev_idx:start_idx])
            para.add_run(batch[start_idx:end_idx]).font.highlight_color = WD_COLOR_INDEX.YELLOW
            prev_idx = end_idx


################################################################
# Main Function

if __name__ == '__main__':
    # Read the transcript in
    directory = "../transcripts/ingested"
    transcript = get_transcript(directory)

    doc = docx.Document()

    # Visualise GATE results
    results = read_results('./gate_results.txt')
    doc.add_heading('GATE', 0)
    write_to_doc(transcript, results)
    
    # Visualise Flair results
    results = read_results('./flair_results.txt')
    doc.add_heading('Flair', 0)
    write_to_doc(transcript, results)

    # Visualise spaCy results
    results = read_results('./spacy_results.txt')
    doc.add_heading('spaCy', 0)
    write_to_doc(transcript, results)

    # Visualise Stanford results
    results = read_results('./stanford_results.txt')
    doc.add_heading('Stanford', 0)
    write_to_doc(transcript, results)

    # Visualise DeepPavlov results
    # TODO

    # Save the document
    doc.save('highlighted_entities.docx')