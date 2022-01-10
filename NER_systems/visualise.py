import docx
from docx.enum.text import WD_COLOR_INDEX
from utilities import get_transcript

################################################################
# Main Function

if __name__ == '__main__':
    # Read the transcript in
    directory = "../transcripts/ingested"
    transcript = get_transcript(directory)

    doc = docx.Document()

    results = list()
    # Read the GATE results in
    with open('./spacy_results.txt') as file:
        line = file.readline()
        entities = list()
        while line != '':
            if line == 'batch_end\n':
                results.append(entities)
                line = file.readline()
                entities = list()
                continue
            entities.append(eval(line))
            line = file.readline()

    # Visualise GATE results
    doc.add_heading('GATE', 0)

    for batch, entities in zip(transcript, results):
        para = doc.add_paragraph()
        prev_idx = 0
        
        for entity in entities:
            start_idx = entity[0][0]
            end_idx = entity[0][1]

            para.add_run(batch[prev_idx:start_idx])
            para.add_run(batch[start_idx:end_idx]).font.highlight_color = WD_COLOR_INDEX.YELLOW
            prev_idx = end_idx
    
    


    # Visualise Flair results
    doc.add_heading('Flair', 0)
    
    para = doc.add_paragraph('''GeeksforGeeks is a Computer Science portal for geeks.''')
    para.add_run(''' It contains well written, well thought and well-explained '''
                ).font.highlight_color = WD_COLOR_INDEX.YELLOW
    para.add_run('''computer science and programming articles, quizzes etc.''')

    
    # Visualise spaCy results
    doc.add_heading('spaCy', 0)
    
    para = doc.add_paragraph('''GeeksforGeeks is a Computer Science portal for geeks.''')
    para.add_run(''' It contains well written, well thought and well-explained '''
                ).font.highlight_color = WD_COLOR_INDEX.YELLOW
    para.add_run('''computer science and programming articles, quizzes etc.''')

    
    # Visualise Stanford results
    doc.add_heading('Stanford', 0)
    
    para = doc.add_paragraph('''GeeksforGeeks is a Computer Science portal for geeks.''')
    para.add_run(''' It contains well written, well thought and well-explained '''
                ).font.highlight_color = WD_COLOR_INDEX.YELLOW
    para.add_run('''computer science and programming articles, quizzes etc.''')
    
    doc.save('highlighted_entities.docx')