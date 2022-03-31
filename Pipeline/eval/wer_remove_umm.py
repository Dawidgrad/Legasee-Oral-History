from jiwer import wer#, cer
from os import listdir
import re
CER = False

#OUTF = 'outputs_hubertall'
OUTF = 'outputsets/wav2vec2_all_lm'
#OUTF = 'outputsets/outputs_hubertall_lm'


def cer(ref, hyp):
    # change ref and hyp to list of characters so that cer can be calculated
    ref = " ".join(list(ref))
    hyp = " ".join(list(hyp))
    return wer(ref, hyp)

wers = []
cers = []

for file in listdir(OUTF):
    if file.endswith('.txt'):
        with open(OUTF + '/' + file, 'r') as f:
            _pred = f.read().upper()
        with open('plain_text/'+file, 'r') as f:
            _true = f.read().upper()
        _pred = _pred.replace('UMM', '').replace('ERR', '').replace('ERM', '').replace("EM", '').replace("ER", '').replace('UM','').replace('AH','').replace('OH','')
        _true = _true.replace('UMM', '').replace('ERR', '').replace('ERM', '').replace("EM", '').replace("ER", '').replace('UM','').replace('AH','').replace('OH','')
        # replace space apostrophe wildcard with apostrophe wildcard i.e you 're -> you're or you 's -> you's or there 's -> there's
        _pred = re.sub(r'\s\'', '\'', _pred)
        _true = re.sub(r'\s\'', '\'', _true)
        # replace apostrophe 
        #_pred = re.sub(r'[^\w\s]','',_pred)
        #_true = re.sub(r'[^\w\s]','',_true)

        wer_ = wer(_true, _pred)
        if CER:
            cer_ = cer(_true, _pred)
        print(f'{file} - {wer_}')
        if CER:
            print(f'{file} - {cer_}\n')
        wers.append(wer_)
        if CER:
            cers.append(cer_)

avg = round(sum(wers)/len(wers), 5)
if CER:
    avg_cer = round(sum(cers)/len(cers), 5)
print(f'Average WER: {avg}')
if CER:
    print(f'Average CER: {avg_cer}')

