from pathlib import Path
import pickle

import librosa
import librosa.display
import numpy as np
import pandas as pd

from tqdm import tqdm_notebook as tqdm

from data_loading_utils import *

import pdb


def load_features(path='', feature_name='log_mel_spec', n_features=40, n_fft=2048,
                  hop_length=512, sample_rate=44100,
                  tmp_path='tmp', tmp_filename=None,
                  files=None, filenames=None, trim=True):

    path, tmp_path = Path(path), Path(tmp_path)
    if not tmp_path.exists():
        tmp_path.mkdir()

    if tmp_filename is None:
        fn_format = '{}_{}_{}_{}_{}_{}_{}_raw.p'
        tmp_filename = tmp_path/fn_format.format(path.name, feature_name,
                                                 n_features, n_fft,
                                                 hop_length, sample_rate, trim)
    tmp_filename = Path(tmp_filename)
    if tmp_filename.exists():
        print('Loading cached data from...', tmp_filename)
        features = pickle.load(open(tmp_filename, 'rb'))
    else:
        if files is None:
            print('Loading audio files...')
            files = load_audio_files(path, filenames, sample_rate=sample_rate, trim=trim)

        features = []
        print('Computing {} features..'.format(feature_name))
        for file in tqdm(files, unit='files'):
            if len(file) == 0:
                print('Empty File')
                feature = np.zeros((n_features, 0))
            elif feature_name == 'mel_spec':
                feature = librosa.feature.melspectrogram(file, sr=sample_rate,
                                                         n_mels=n_features, n_fft=n_fft,
                                                         hop_length=hop_length)
            elif feature_name == 'log_mel_spec':
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

            features.append(feature)

        print('Saving data..')
        pickle.dump(features, open(tmp_filename, 'wb'))

    print('Loaded features for {} files'.format(len(features)))
    return features
