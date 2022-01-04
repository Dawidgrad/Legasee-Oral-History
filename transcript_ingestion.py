# Utility functions and predefined transformations for ingestion of Legasee transcript files

# PyMuPDF - https://pypi.org/project/PyMuPDF/
import fitz

# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
from bs4 import BeautifulSoup, Tag
import bs4

from IPython.core.display import display, HTML
import re
from datetime import datetime, timedelta
import pandas as pd

######

# Various RegEx patterns used in logic

## Standard PDF format
patt_wis = re.compile(r"what is said",re.I)
patt_ts  = re.compile(r"\d{1,2}:\d{2}:\d{2}")
patt_sof = re.compile(r"(start|end) of film(s)?(\s+\d+)?",re.I)

### Sometimes get timestamp and text in a single span - need to split them
patt_joined  = re.compile(r"(?P<stamp>\d{1,2}:\d{2}:\d{2})(?P<after_stamp>.+)")


## Fancy PDF format
patt_name = re.compile(r"(?P<speaker>[^\d\W]+(\s+[^\d\W]+)*)\s?:(?P<content>.*)")
patt_foot = re.compile(r"File name:")
patt_eoa  = re.compile(r"\[(?P<stamp>\d{1,2}:\d{2}:\d{2})\]\s*\[END OF AUDIO\]",re.I)


## Text file
patt_ts  = re.compile(r"\d{1,2}:\d{2}:\d{2}")
patt_sof = re.compile(r"[*]+ start of film \d+",re.I)

######

# Import functions

## Standard PDF page
def page_to_ts(docpage):
    ''' Extract timestamp, speaker identity and text from PDF transcript from Legasee archive.
    Assumptions:
        - Input is a page from a PDF file, ingested with fitz (PyMuPDF)
        - Format is the one most frequently given to us:
            - Main content is in a table, with (repeating) headers "Time Code" and "What is Said"
            - Header and footer, if present, may be ignored, along with any preamble (e.g. metadata about date of transcription)
                - Header and footer end up appearing before the content in the recovered HTML document
            - Bold text indicates interviewer questions, non-bold interviewee. "Start of Film X" treated as distinct speaker.
                - No sections of bold appear in the interviewee's text
            
            - Line breaks in recovered text are not retained
            - Need to do something about the markers for unintelligible content within the text - they could also let us subdivide to get more stamps
    '''
    
    ts_content = []
    _started = 0
    _last_time = None
    _last_speaker = ''
    
    _block = [None, '', []]

    
    # Utility function for block handling. When a new block is started, append current block if it has any content before initiating a new one
    def new_block(time,speaker=''):
        
        nonlocal _block, ts_content
        
        if len(_block[2]):
            # Concat to string, replace line breaks with spaces, compress whitespace
            _block[2] = re.sub("\s{2,}"," ",re.sub("\n"," "," ".join(_block[2])))
            ts_content.append(tuple(_block))

        _block = [time,speaker,[]]
        #print("New block created")

    
    def process_para(p,speaker_type):
        #print(" Processing paragraph")
        
        nonlocal _block, _last_time, _last_speaker, ts_content, _started
        
        # Process strings and tag elements differently
        if type(p) == bs4.element.Tag:
            pt  = p.text
            pct = p.text.strip(' \t\n')
            
        elif type(p) == str:
            pt = p
            pct = p.strip(' \t\n')
        
        
        # After finding the start of the content, we want to tag speaker and timestamp as well as content, as a triple
        if _started == 1:
            
            # If speaker type has changed, start a new block
            if speaker_type != _last_speaker:
                _last_speaker = speaker_type
                new_block(_last_time,speaker_type)

            # Sometimes get timestamp and text in a single span - need to split them
            join = re.fullmatch(patt_joined,pct)
            if join:
                #print("  Joined section")
                process_para(join.group('stamp'),speaker_type)
                process_para(join.group('after_stamp'),speaker_type)

            
            # "Start of film X" or "end of film(s)" triggers specific treatment - start a new block, unless current block only has a timestamp
            elif re.fullmatch(patt_sof,pct):
                if len(_block[2]):
                    new_block(None,speaker='New Film')
                else:
                    _block[1] = 'New Film'
                    
                _block[2] = ['New Film']
                new_block(_last_time)
                
            
            # Paragraph is a timestamp - new block
            elif re.fullmatch(patt_ts,pct):
                t = datetime.strptime(pct,"%H:%M:%S")
                delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                # Update most recent timestamp in case needed elsewhere
                _last_time = delta
                
                new_block(delta,speaker_type)      
                    
                
            # Otherwise - general text
            else:
                _block[2].append(pt)
                
            
        # Until first "What is Said" encountered (and including that para), don't care about the text
        if re.fullmatch(patt_wis,pct):
            _started = 1
            
            
    
    raw_html = docpage.get_text("html")
    page_bs = BeautifulSoup(raw_html)
    
    # Get paragraphs from content, as a list
    paras = page_bs.find_all("p")
    
    for p in paras:

        # If paragraph contains bold sections, need to process those as subunits
        if p.find("b"):
            for b in p.find_all("b"):
                process_para(b,'Interviewer')
        
            # End block after bolds, as some files do not include new timestamps for the user
            #new_block(_last_time,'Interviewee')
        else:
            process_para(p,'Interviewee')
            
    # Append final block if it has any content
    new_block(None)
        
    return ts_content


## Fancy PDF page
def fancy_page_to_ts(page):
    
    ts_content = []
    _started = 1
    
    _last_speaker = ''
    
    _block = [None, '(cont.)', []]
    
    
    # Utility function for block handling. When a new block is started, append current block if it has any content before initiating a new one
    def new_block(time,speaker=''):
        
        nonlocal _block, ts_content
        
        if len(_block[2]):
            # Concat to string, replace line breaks with spaces, compress whitespace
            _block[2] = re.sub("\s{2,}"," ",re.sub("\n"," "," ".join(_block[2])))
            ts_content.append(tuple(_block))

        _block = [time,speaker,[]]
        #print("New block created")

        
    def process_para(p):
        nonlocal _block, _last_speaker, ts_content, _started
        
        if _started:
            
            pt = p.text
            
            # Footer appears later in page content - need to reset _started when we encounter "File name:"
            # Note that this matches the speaker name pattern (though won't appear in a bold tag)
            if patt_foot.match(pt):
                _started = 0
                
            # End of audio marker
            elif patt_eoa.match(pt):
                
                t = datetime.strptime(patt_eoa.match(pt).group('stamp'),"%H:%M:%S")
                delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                
                new_block(delta,speaker='End of audio')
                _block[2] = ['End of audio']
            
            else:
                # If the first bold item looks like "Name: " then new block, set speaker
                if p.find("b"):
                    btext = p.find("b").text.strip(' \t\n')

                    nmatch = patt_name.match(btext)
                    if nmatch:
                        _last_speaker = nmatch.group('speaker')
                        new_block(None,speaker=nmatch.group('speaker'))


                # If pt begins with current speaker, remove that from the beginning
                if pt.startswith(_last_speaker+':'):
                    pt = pt[len(_last_speaker)+1:]

                _block[2].append(pt)
                      
            
    raw_html = page.get_text("html")
    page_bs = BeautifulSoup(raw_html)
    
    # Get paragraphs from content, as a list
    paras = page_bs.find_all("p")
    
    for p in paras:
        process_para(p)
            
    # Append final block if it has any content
    new_block(None)
        
    return ts_content


## Text file
def text_to_ts(filename):
    
    # Utility function for block handling. When a new block is started, append current block if it has any content before initiating a new one
    def new_block(time,speaker=''):
        
        nonlocal _block, ts_content
        
        if len(_block[2]):
            # Concat to string, replace line breaks with spaces, compress whitespace
            _block[2] = re.sub("\s{2,}"," ",re.sub("\n"," "," ".join(_block[2])))
            ts_content.append(tuple(_block))

        _block = [time,speaker,[]]
        #print("New block created")
        
    
    with open('./raw/'+filename+'.txt') as ofile:
        
        _i = 0
        _started = 0
        
        ts_content = []
        _block = [None, '', []]
        
        
        for line in ofile.readlines():
            
            line = line.strip(' \t\n')
            
            # Start of film X
            if patt_sof.fullmatch(line):
                _started = 1
                
                new_block(None,speaker='New Film')
                _block[2] = ['New Film']
            
            if _started:
                
                # Timestamp
                if patt_ts.fullmatch(line):
                    
                    t = datetime.strptime(line,"%H:%M:%S")
                    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                
                    new_block(delta,speaker='')
                
            
                else:
                    _block[2].append(line)
                

        new_block(None)
        
        return ts_content