from diarize.spectral_clustering import spectral_cluster
import malaya_speech
import numpy as np
import soundfile as sf
from scipy import interpolate


def resample(data, old_samplerate, new_samplerate):
    """
    Resample signal.
    Parameters
    ----------
    data: np.array
    old_samplerate: int
        old sample rate.
    new_samplerate: int
        new sample rate.
    Returns
    -------
    result: data
    """
    old_audio = data
    duration = data.shape[0] / old_samplerate
    time_old = np.linspace(0, duration, old_audio.shape[0])
    time_new = np.linspace(
        0, duration, int(old_audio.shape[0] * new_samplerate / old_samplerate)
    )

    interpolator = interpolate.interp1d(time_old, old_audio.T)
    data = interpolator(time_new).T
    return data


def read_audio(data, old_samplerate, sample_rate=22050):
    if len(data.shape) == 2:
        data = data[:, 0]

    if old_samplerate != sample_rate and sample_rate is not None:
        data = resample(data, old_samplerate, sample_rate)
    else:
        sample_rate = old_samplerate

    return data, sample_rate


def load(file: str, sr=16000, scale: bool = True):
    """
    Read sound file, any format supported by soundfile.read
    Parameters
    ----------
    file: str
    sr: int, (default=16000)
        new sample rate. If input sample rate is not same, will resample automatically.
    scale: bool, (default=True)
        Scale to -1 and 1.
    Returns
    -------
    result: (y, sr)
    """
    data, old_samplerate = sf.read(file)
    y, sr = read_audio(data, old_samplerate, sr)
    if scale:
        y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr


def load_models(y_int):
    return malaya_speech.speaker_vector.deep_model('speakernet'), malaya_speech.vad.webrtc(minimum_amplitude = int(np.quantile(np.abs(y_int), 0.2)))


def grouped_VAD_frames(y_int, fs, vad):
    frames_int = list(malaya_speech.utils.generator.frames(y_int, 30, fs, append_ending_trail=False))
    frames_webrtc = [(frame, vad(frame)) for frame in frames_int]
    grouped_vad = malaya_speech.utils.group.group_frames(frames_webrtc)
    grouped_vad = malaya_speech.utils.group.group_frames_threshold(grouped_vad, threshold_to_stop=0.05)
    return grouped_vad


def get_speaker_turns(_speaker_stamps):
  # min stamp duration is 0.5 second
  speaker_stamps = [el for el in _speaker_stamps if el[1] != 'not a speaker' and el.duration > 0.5]
  timestamps = lambda x: [x.timestamp, x.duration+x.timestamp]
  turn_list = [{'speaker':speaker_stamps[0][1], 'start':0, 'end': timestamps(speaker_stamps[0][0])[1]}]

  for frame, speaker in speaker_stamps[1:]:
    start, end = timestamps(frame)
    if turn_list[-1]['speaker'] != speaker:
      turn_list.append({
          'speaker':speaker,
          'start':start,
          'end':end
      })
    else:
      turn_list[-1]['end'] = end
  
  turn_list = [turn for turn in turn_list if turn['end'] - turn['start'] > 0.5]
  return turn_list


def run_diarization(file=None, wav=None):
    if file is not None:
        audio, fs = load(file)
    else:
        audio, fs = wav, 16000
    # Load the speaker vector model and the webrtc VAD model
    y_int = malaya_speech.astype.float_to_int(audio)
    model_speakernet, vad = load_models(y_int)

    # Split the audio sample by a given time plit and generate the required grouped VAD frames
    grouped_vad = grouped_VAD_frames(y_int, fs, vad)

    # Run speaker diarization
    result_diarization_sc_speakernet = spectral_cluster(grouped_vad, model_speakernet, min_clusters=2, max_clusters=2)

    # Obtain diarization frames and the output speaker timestamp list
    grouped_frames = malaya_speech.group.group_frames(result_diarization_sc_speakernet)
    timestamp_list = get_speaker_turns(grouped_frames)

    return timestamp_list



def select_speaker(turns, output):
    '''
    turns = [{'speaker':speaker, 'start':start, 'end':end}, ...] list of speaker turns \n
    output = [{'text':text, 'start':start, 'end':end}, ...] list of text with timestamp \n
    this function will assign the speaker of each text based on the speaker turns
    '''
    diarized_output = []
    for word in output:
   
        nearest_turn = {
            'speaker':None,
            'dist':float('inf')
        }
        for i, turn in enumerate(turns):
            dist = min(abs(word['start'] - turn['start']), abs(word['end'] - turn['end']))
            if dist < nearest_turn['dist']:
                nearest_turn['dist'] = dist
                nearest_turn['speaker'] = turn['speaker']
            
        diarized_output.append({
            'text': word['text'],
            'start': word['start'],
            'end': word['end'],
            'speaker': nearest_turn['speaker']
        })
    
    return [diarized_output, turns]








'''
def get_speaker_turns(_speaker_stamps):
  speaker_stamps = [el for el in _speaker_stamps if el[1] != 'not a speaker']
  timestamps = lambda x: [x.timestamp, x.duration+x.timestamp]
  turn_list = [{'speaker':speaker_stamps[0][1], 'start':0, 'end': timestamps(speaker_stamps[0][0])[1]}]

  for frame, speaker in speaker_stamps[1:]:
    #if frame.duration > 0.25:
    start, end = timestamps(frame)
        #if turn_list[-1]['speaker'] != speaker:
    turn_list.append({
        'speaker':speaker,
        'start':start,
        'end':end
    })
        #else:
            #turn_list[-1]['end'] = end
  
  #turn_list = [turn for turn in turn_list if turn['end'] - turn['start'] > 0.5]
  return turn_list









def select_speaker(turns, output):
    
    turns = [{'speaker':speaker, 'start':start, 'end':end}, ...] list of speaker turns \n
    output = [{'text':text, 'start':start, 'end':end}, ...] list of text with timestamp \n
    this function will assign the speaker of each text based on the speaker turns
    
    diarized_output = []
    for word in output:
   
        total_overlap = {} # {speaker: overlap, ...}
        for i, turn in enumerate(turns):
            if word['start'] >= turn['start'] and word['end'] <= turn['end']:
                overlap = turn['end'] - word['start']
            elif word['end'] <= turn['end'] and word['start'] >= turn['start']:
                overlap = word['end'] - turn['start']
            elif word['start'] <= turn['start'] and word['end'] >= turn['end']:
                overlap = turn['end'] - turn['start']
            else:
                overlap = 0
            total_overlap[turn['speaker']] = overlap if turn['speaker'] not in total_overlap else total_overlap[turn['speaker']] + overlap
            #total_overlap[turn['speaker']] +=
            
        if all(total_overlap.values()) == 0:
            min_idx = 0
            min_dist = float('inf')
            for i, turn in enumerate(turns):
                dist = min( abs( turn['end'] - word['start'] ) , abs( word['end'] - turn['start'] ) )
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            diarized_output.append({
                'text':word['text'],
                'start':word['start'],
                'end':word['end'],
                'speaker':turns[min_idx]['speaker']
            })
        else:
            diarized_output.append({
                'text': word['text'],
                'start': word['start'],
                'end': word['end'],
                'speaker': list(total_overlap.keys())[np.argmax(list(total_overlap.values()))][np.argmax(total_overlap.values())]
            })
    
    return [diarized_output, turns]

'''