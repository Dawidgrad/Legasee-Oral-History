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

    # Read the GATE results in


    # Visualise GATE results
    doc.add_heading('GATE', 0)

    for batch in transcript:
        # 
        print()
    
    para = doc.add_paragraph('''GeeksforGeeks is a Computer Science portal for geeks.''')
    para.add_run(''' It contains well written, well thought and well-explained '''
                ).font.highlight_color = WD_COLOR_INDEX.YELLOW
    para.add_run('''computer science and programming articles, quizzes etc.''')


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