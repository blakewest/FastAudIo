from fastai.imports import *
from fastai.layer_optimizer import *
from enum import IntEnum
#from fastai.transorms import *


#return raw audio at specified length
def adj_length(raw, length=3*44100): 
    raw_len = len(raw)
    if raw_len < length:
        raw = np.pad(raw, ((length-raw_len)//2), 'constant')
    raw = raw[:length]
    raw_max = np.argmax(raw)
    start = max(0, (raw_max-(length//2)))
    end = start+(raw_max+(length//2))
    if start == 0:
        end = start+length
    if start+end > length:
        start = 0
        end = length
    return raw[start:end]


n_mels=40
n_fft=2048  
hop_length=512
sr=44100

def get_mel(raw, sr=sr, feature_name='log_mel_spec', n_mels=n_mels, 
            n_fft=n_fft, hop_length=hop_length):
    if feature_name == 'mel_spec':
        feature = librosa.feature.melspectrogram(file, sr=sample_rate, 
                                                 n_mels=n_features, n_fft=n_fft, 
                                                 hop_length=hop_length)
    elif feature_name == 'leg_mel_spec':
        mel_spec = librosa.feature.melspectrogram(file, sr=sample_rate, 
                                                  n_mels=n_features, n_fft=n_fft, 
                                                  hop_length=hop_length)
        log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
        # amplitude_to_db normalizes to [-80.0, 0.0]
        feature = (log_mel_spec + 40.0) / 40.0  # rescale to [-1.0, 1.0]
    elif feature_name == 'mfcc':
        feature = librosa.feature.mfcc(file, sr=sample_rate, 
                                       n_mfcc=n_features, n_fft=n_fft, 
                                       hop_length=hop_length)
    else:
        raise NotImplementedError
    
    return feature
