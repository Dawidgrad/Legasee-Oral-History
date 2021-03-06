import json
import re
from enum import Enum

class TranscriptType(Enum):
    ANNOTATION = 1
    OUTPUT = 2

# Get the appropriate transcript based on the type of input we want to use
def get_transcripts(type, input_path):
    result = []
    
    if type == TranscriptType.ANNOTATION:
        # Get the json file
        with open(input_path, 'r') as json_file:
            json_list = list(json_file)
        
        raw_transcripts = []
        # Extract the raw transcripts from json file
        for json_str in json_list:
            document = json.loads(json_str)
            raw_transcripts.append(document['data'])
        
        result = segment_trascripts(raw_transcripts)
    
    elif type == TranscriptType.OUTPUT:
        # Get the output file from previous step of the pipeline
        with open(input_path, "r") as file:
            file_content = file.read()

        # Ensure the string is not too large
        segment_size = 500
        result = segment_string(segment_size, file_content)

    return result

def segment_trascripts(raw_transcripts):
    result = []
    timestamp_regex = r'(\d{2}:\d{2}:\d{2})'

    # Split transcripts into segments
    for transcript in raw_transcripts:
        split_transcript = re.split(timestamp_regex, transcript)
        split_transcript = split_transcript[3:]

        # Create a dictionary of timestamp keys and transcript text value
        transcript_dict = dict()

        for i in range(0, len(split_transcript), 2):
            transcript_dict[split_transcript[i]] = split_transcript[i + 1]

        # Split the transcript into similar segment sizes
        segment_size = 500
        segmented_dict = dict()
        for key, value in transcript_dict.items():
            segment_list = segment_string(segment_size, value)
            segmented_dict[key] = segment_list
            
        result.append(segmented_dict)

    return result

def segment_string(segment_size, text):
    segmented = []
    idx = 0
    while True:
        # Find how far is the closest full stop (relative to segment size)
        idx_limit = idx + segment_size if (idx + segment_size) < len(text) else (len(text))
        full_stop_idx = text[:idx_limit].rfind('.')
        triple_dot_idx = text[:idx_limit].rfind('???')
        
        end_idx = full_stop_idx if full_stop_idx > triple_dot_idx else triple_dot_idx

        # What to do if no full_stop can be found
        if len(text) < segment_size and end_idx <= idx:
            end_idx = len(text) - 1
        elif end_idx <= idx:
            end_idx = text[:idx + int(segment_size / 2)].rfind(' ')

        # Get the segment from current index to closest full stop
        segment = text[idx:end_idx + 1]

        # Preprocess the segment and add it to the output dictionary
        segment = preprocess_segment(segment)

        if (len(segment) > 5):
            segmented.append(segment)
        idx = end_idx + 1
        
        # Check if the last segment has been encountered already
        if (idx > (len(text) - segment_size)):
            last_segment = text[end_idx:]
            last_segment = preprocess_segment(last_segment)
            if (len(last_segment) > 5):
                segmented.append(last_segment)
            break
        
    return segmented

def preprocess_segment(segment):
    # Remove newlines
    preprocessed_segment = segment.replace('\n', ' ')

    # Remove "Start of Film" and "End of Film" text
    pattern_start = r'\*\* Start of Film [0-9]*'
    pattern_end = r'End of F(i|I)lms'
    preprocessed_segment = re.sub(pattern_start, '', preprocessed_segment)
    preprocessed_segment = re.sub(pattern_end, '', preprocessed_segment)

    # Remove whitespaces
    preprocessed_segment = preprocessed_segment.strip()

    return preprocessed_segment

def tag_transcripts(entities, transcripts):
    segment_idx = 0
    transcripts_idx = 0
    segment_entities = []
    tagged_output = ['']

    for entity in entities:
        if entity != 'segment_end' and entity != 'transcript_end':
            segment_entities.append(entity)

        if entity == 'segment_end':
            tagged_segment = tag_segment(transcripts[transcripts_idx][segment_idx], segment_entities)
            tagged_output[transcripts_idx] = ' '.join([tagged_output[transcripts_idx], tagged_segment])
            segment_entities = []
            segment_idx += 1    
        elif entity == 'transcript_end':
            transcripts_idx += 1
            segment_idx = 0
            tagged_output.append('')

    return tagged_output

def tag_segment(text, label):
    # Rewrites text adding the tags without taking embedded tags into account 
    transcript = ''
    previous_idx = 0
    for NER in label:
        pre = NER[0][0]
        post = NER[0][1]
        tag = NER[1]
        pretag = '<' + tag + '>'
        posttag = '<\\' + tag + '>'
        transcript += text[previous_idx:pre] + pretag + text[pre:post] + posttag
        previous_idx = post
    transcript += text[previous_idx:]

    # Looks at embedded tags:<Title>Captain<\Title><Person>James<\Person> --> <Person><Title>Captain<\Title>James<\Person>
    pattern = r'<([\w ]+)>([\w ]+)<\\([\w ]+)><([\w ]+)>([\w ]+)<\\([\w ]+)>'
    for a in re.finditer(pattern, transcript):
        if a.group(2) in a.group(5):
            new_tag = a.group(5).replace(a.group(2)+' ', '')
            new_pattern = rf'<\4><\1>\2<\\\3>{new_tag}<\\\6>'
            transcript = re.sub(pattern, new_pattern, transcript, count = 1)

        elif a.group(5) in a.group(2):
            new_tag = a.group(2).strip(a.group(5)+' ')
            new_pattern = rf'<\4><\1>{new_tag}<\\\3>\5<\\\6>'
            transcript = re.sub(pattern, new_pattern, transcript, count = 1)
        
    return transcript

def write_to_file(directory, data):
    with open(directory, "w") as file:
        for item in data:
            file.write(str(item) + '\n')