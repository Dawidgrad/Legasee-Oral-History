# Add speaker turn to each word from diarisation of speakers pickle file
import pickle
import os
import argparse
from typing import Dict

def speaker_turn(x:Dict) -> Dict:
  interviewer = []
  interviewee = []
  for i, turn in enumerate(x['Output']['speaker_turns']):
    start, end = turn['start'] if i != 0 else 0, turn['end'] # first speaker turn should always start at 0
    if turn['speaker'] == 'Interviewer':
      interviewer.append([start, end])
    else:
      interviewee.append([start, end])

  for word_dict in x['Output']['segmented_output']:
    mid = (word_dict['start'] + word_dict['end'])/2

    word_dict['Speaker_turns'] = 'Interviewee'
    for intervaler in interviewer:
      if mid >= intervaler[0] and mid <= intervaler[1]:
        word_dict['Speaker_turns'] = 'Interviewer'
        break
  # return x['Output']['segmented_output']
  return x
