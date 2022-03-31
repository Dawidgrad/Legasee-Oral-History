import pandas as pd
from os import listdir as ls


audio_dir = './audio/'
txt_dir = './plain_text/'

datadict = {
    'Text_File': [],
    'Wav_File': [],
}
txts = ls(txt_dir)

for file in ls(audio_dir):
    if file.endswith('.wav'):
        datadict['Wav_File'].append(file)
        if (file[:-4] + '.txt') in txts:
            datadict['Text_File'].append(file[:-4] + '.txt')
        else:
            datadict['Text_File'].append(input('Enter text file for ' + file + ': '))


csv = pd.DataFrame(datadict)
csv.to_csv('data.csv', index=False)
