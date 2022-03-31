# This directory includes all the code to run the transcription and extraction pipeline
- The ASR model is initiated by running 'python main.py', once this is complete an intermediatery output is saved in the cache folder. To initiate downstream tasks, and produce a complete output run 'python main.py --downstream -ner --diarization --punctuation' downstream tasks are toggled on by providing them as an argument, outputs are saved in the output folder.
- All inputted audio files should be formatted as 16khz wav files. The audio that requires processing should be listed in a csv file in the column 'Wav_Files', this csv file is passed to the pipeline using the argument '--csv'.
- Full details for each command line argument can be seen by running 'python main.py --help'
- To install the relevant packages use the requirements.txt file 
