# import subprocess
import os
    
def get_named_entities(text, base_dir):
    # Write string to a file in input directory
    with open(f"{base_dir}/input/input_string.txt", "w") as file:
        file.write(str(text))

    # Run NER library/ies
    os.system(f'python {base_dir}/spacy_ner.py -d {base_dir} -o')

    # Read output file into tagged_text
    with open(f'{base_dir}/ner_output/spacy_tagged_transcript.txt') as file:
        tagged_text = file.read()

    return tagged_text