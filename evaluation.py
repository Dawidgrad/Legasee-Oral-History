# Utility functions and predefined transformations for evaluation of 
# ASR system performance on the Legasee project.

import pandas as pd
import jiwer
import re
import os

import nltk
from nltk.corpus import stopwords

from typing import Union, List, Mapping

# Needs to have been run to be used in functions
# nltk.download('stopwords')

# Used to convert numbers to words, e.g. 8 o'clock -> eight o'clock, 1944 -> nineteen forty four
import inflect
# Inflect is more flexible, but doesn't create ordinals as words - use num2words for that
from num2words import num2words

# Overwriting current jiwer version to enable weighted calculations
from measures import compute_measures



# Custom transformations for handling digits in gold transcripts
class DigitsToWords(jiwer.AbstractTransform):
    def __init__(self, target=re.compile(r"\b[,.\d]+\b"), **ntw_opts):
        """
        Use inflect library's number_to_words functionality to substitute digits for corresponding strings.
        E.g. 8 o'clock -> eight o'clock.
        Note that some instances may warrant distinct treatment, e.g. $1,000 might have desired output "one thousand dollars", 
         "1944" may want to be "nineteen fourty-four".
        To enable this, only substrings matching the `target` compiled regex pattern are processed. By default, this is any
         "word" (per standard RegEx word boundaries) consisting of digits, commas and decimal points.
         
        **kwargs are passed through to the number_to_words function - see docs at https://pypi.org/project/inflect/
        Note that passing the groups parameter introduces additional commas.
        """
        self.ie = inflect.engine()
        
        self.target = target
        self.ntw_opts = ntw_opts
        
    def process_string(self, s: str, **ntw_opts):
        for m in re.finditer(self.target, s):
            repl = self.ie.number_to_words(m[0], **self.ntw_opts)
            s = s.replace(m[0],repl,1)
        return s

    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]
    
    
    
class OrdinalsToWords(jiwer.AbstractTransform):
    def __init__(self, target=re.compile(r"\b(?P<numpart>[,.\d]+)(st|nd|rd|th)\b")):
        """
        Use num2words library's functionality to substitute ordinals for corresponding strings.
        E.g. 22nd -> twenty-second.
        https://pypi.org/project/num2words/
        
        If modifying the target regex, note that the group label 'numpart' is required 
         (for the numeric section which is retained and converted to ordinal words).
        """
        self.target = target
        
    def process_string(self, s: str):
        for m in re.finditer(self.target, s):
            repl = num2words(m['numpart'], ordinal=True)
            # Want to replace entire match, not just the digits (or we get "secondnd")
            s = s.replace(m[0],repl,1)
        return s
    
    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]
    
    
### Transformations, for use at different evaluation stages

# Etc. abbreviations expanded
etcdict = {'etcetera' : 'et cetera',
           'etc' : 'et cetera',
          }

# Hesitations - removed
hesdict = {r'\b[Uu]m+\b' : '',
           r'\b[Ee]r+(m+?)\b' : '',
           r'\b[Uu]h+\b' : '',
            }

transform_baseline = jiwer.Compose([
    jiwer.Strip(),
    
    # Similar to jiwer.RemoveKaldiNonWords() but also removing parentheticals e.g. "(unclear)"
    # NB: need non-greedy match in the middle!
    jiwer.SubstituteRegexes({r"[\[{\(\<].*?[\]}\)\>]": r" ",}),
    
    # Year processing - convert 200x to "two thousand and x"
    DigitsToWords(target = re.compile(r"\b200\d{1}\b"), group=0),
    # Other years (from 1700 on) - convert e.g. 1984 to "nineteen, eighty-four"
    # NB: place prior to removal of punctuation as introduces new commas
    DigitsToWords(target = re.compile(r"\b(17|18|19|20)\d{2}\b"), group=2, zero='oh'),
    
    # Ordinals - "1st" -> "first"
    OrdinalsToWords(),
    
    # Other numbers (as standalone words)
    # NB: things like "£1000" unchanged - might want to revisit
    DigitsToWords(),
    
    jiwer.ToLowerCase(),

    # Extra fix for common typo
    jiwer.SubstituteWords({'im' : "i'm", "Im" : "I'm"}),
    
    # E.g. "I've" -> "I have"
    # "I'd" -> "I would" (& similar) causes errors when it should be "I had"
    #jiwer.ExpandCommonEnglishContractions(),

    # etcetera, etc. -> et cetera
    jiwer.SubstituteWords(etcdict),
    
    # Hesitations
    jiwer.SubstituteRegexes(hesdict),
    
    # Hyphens appear between words; sub for spaces rather than just deleting
    jiwer.SubstituteRegexes({'[-_]':' '}),
    
    # jiwer transform uses string.punctuation, which doesn't include e.g. ’ - replace with ditching RegEx non-words
    #jiwer.RemovePunctuation(),
    jiwer.SubstituteRegexes({'[^\w\s]':''}),
    
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveEmptyStrings(),
    jiwer.Strip(),
    jiwer.SentencesToListOfWords(word_delimiter=" "),
]) 


### Keyword preparation
# For all_words, want to:
# - strip e.g. parentheses
# - drop stopwords
# - drop "St / St."
# - drop if all numeric?
# - drop ordinals (keep for individual, drop for large list)
# - drop empty string

# - do something with things like O'Brien, Women's, Linton-on-Ouse?
#  -- transform using the same transformation as used in evaluation; though could specify a different one

def kword_prep(inlist,transform,drop_ordinal=True):
    
    outlist = []

    if drop_ordinal:
        npatt = re.compile(r"[,.\d]+(st|nd|rd|th)?")
    else:
        npatt = re.compile(r"[,.\d]+")
                
    for s in inlist:
    
        s = s.strip(' ()[]')
            
        if s.lower() in stopwords.words('english'):
            pass

        elif s.lower() in ['st', 'st.']:
            pass

        elif re.fullmatch(npatt,s):
            pass

        elif s == '':
            pass

        elif len(transform(s)) > 1:
            outlist.extend(kword_prep(transform(s),transform))

        else:
            outlist.extend(transform(s))
    
    return set(outlist)


### Score for an individual interviewee (with a given model). Returns a dataframe with several metrics.
def score_name(name,
               meta_frame,
               syspath,
               model_folder,
               test_train,
               key_weight,
               shared_dict,
               transform,
               in_folders = False, # If true, attempt to read system outputs from multiple files within a subfolder of model folder. 
                                   # Otherwise, grab a single file from within model folder.
              ):
    
    # Do lookup in meta_df, confirm get 1 match
    _name_space = re.sub('_', ' ',name)

    name_meta = meta_frame[meta_frame.Title == _name_space].reset_index()

    _metafound = len(name_meta)

    if _metafound != 1:
        raise ValueError('{} entries found in metadata sheet for {}. Review metadata.'.format(str(_metafound),_name_space))

    # Confirm transcript available - else return None ?
    if name_meta.Transcript[0] < 1:
        return None

    # Get gold transcript
    _gfolder = os.path.expanduser("~")+syspath+'/data/legasee/'+test_train+'/transcripts'
    gold_ts = _read_gold_transcript(_gfolder,name)

    
    # Get system transcripts (potentially from multiple files)
    _mfolder = os.path.expanduser("~")+syspath+'/system_outputs/'+model_folder
    
    if in_folders:
        sys_ts = _read_sys_transcripts(_mfolder+'/'+name)

    else:
        sys_ts = _read_sys_file(_mfolder+'/'+name+'.txt')
    
    
    # Get individual keywords
    key_words = name_meta['Priority Words'][0].copy()
    key_words.extend(name_meta['Name Words'][0])

    key_words_clean = kword_prep(key_words, transform)

    kdict = {k : key_weight for k in key_words_clean}
    k1dict = {k : 1 for k in key_words_clean}
    
    # Mixed dictionary
    comb_dict = shared_dict.copy()
    comb_dict.update(kdict)

    ## Metrics
    # Flatten lists of text for comparison
    sys_text = ' '.join(sys_ts)
    gold_text = ' '.join(gold_ts)
    
    # want the following (with parameterised weights):

    # - straight WER/WIP (complete xscript)
    # - WWER / WIP(just own tagwords)
    # - WWER / WIP (own + all tagwords)
    # - KWER / KWIP (just own tagwords)

    unw_measures = compute_measures(gold_text, sys_text,
            truth_transform=transform, 
            hypothesis_transform=transform)

    m_df = _make_measure_frame(unw_measures,_name_space,'Unweighted')
    
    wwer_self_measures = compute_measures(gold_text, sys_text,
                truth_transform=transform, 
                hypothesis_transform=transform,
                weights = kdict)
    
    ww_s_df = _make_measure_frame(wwer_self_measures,_name_space,'Weighted (own keywords)')
    
    wwer_all_measures = compute_measures(gold_text, sys_text,
                truth_transform=transform, 
                hypothesis_transform=transform,
                weights = comb_dict)
    
    ww_a_df = _make_measure_frame(wwer_all_measures,_name_space,'Weighted (own + shared keywords)')
    
    kwer_measures = compute_measures(gold_text, sys_text,
                truth_transform=transform, 
                hypothesis_transform=transform,
                weights = k1dict,
                default_weight = 0)
    
    kw_df = _make_measure_frame(kwer_measures,_name_space,'Keywords (own) only')
    
    # Keyword incidence
    kw_incidence = []

    for kw in kdict.keys():
        kw_incidence.extend([(kw,transform(gold_text).count(kw))])
    
    # Combine into score frame
    m_df['Keywords','Incidence'] = [kw_incidence]
    m_df = m_df.sort_index(axis=1)
    
    scores_df = m_df.join(ww_s_df).join(ww_a_df).join(kw_df)

    # Output score frame, text
    return scores_df, sys_text, gold_text


### Transcript readers
def _read_sys_folder(sys_folder):

    sys_ts = []
    
    # For single files with "-- NEW VIDEO --" markers
    patt_newvid = re.compile(r"\-+\s+NEW VIDEO\s+\-+")
    
    f_list = os.listdir(sys_folder)
    # Sort file list
    f_list.sort()
    
    for fname in f_list:
        
        # Only interested in .txt files
        if fname[-4:] == '.txt':
           
            sys_text = ''
            with open(sys_folder+'/'+fname,'r') as sysin:
                for l in sysin.readlines():

                    if re.fullmatch(patt_newvid,l.strip()):
                        sys_ts.append(sys_text.strip())
                        sys_text = ''

                    else:
                        sys_text = " ".join([sys_text,l.strip()])

                if len(sys_text.strip()):
                    sys_ts.append(sys_text.strip())
                    

    return sys_ts


def _read_sys_file(sys_file):

    sys_ts = []
            
    # Only interested in .txt files
    if sys_file[-4:] == '.txt':

        sys_text = ''
        with open(sys_file,'r') as sysin:
            for l in sysin.readlines():
                
                sys_text = " ".join([sys_text,l.strip()])

                if len(sys_text.strip()):
                    sys_ts.append(sys_text.strip())

        return sys_ts
    
    else:
        raise NameError('Input {} not valid - expected a .txt file'.format(sys_file))


def _read_gold_transcript(gold_folder,name):
    
    patt_marker = re.compile(r"(\-+\s+NEW VIDEO\s+\-+)|(\s*New\s+Film\s*)|(\s*Start\s+of\s+Film(\s*\d+)?\s*)|(\s*End\s+of\s+Films?\s*)",re.I)
    
    gold_df = pd.read_csv(gold_folder+'/'+name+'.tsv', delimiter='\t', index_col=0)

    hum_ts = []

    hum_text = ''

    for l in gold_df.Transcript:
        if re.fullmatch(patt_marker,l.strip()):
            if len(hum_text.strip()):
                hum_ts.append(hum_text.strip())
            hum_text = ''

        else:
            hum_text = " ".join([hum_text,l.strip()])
        
    if len(hum_text.strip()):
        hum_ts.append(hum_text.strip())
            
    return hum_ts


### Convert metrics to dataframe
def _make_measure_frame(measures,name,i_label):

    m_df = pd.DataFrame.from_dict(measures, orient='index',
                                  columns=[name]
                                 ).transpose().drop('wil',axis=1)
    # Add additional level to the index
    m_df.columns = pd.MultiIndex.from_product([[i_label], m_df.columns])
    
    return m_df