from pathlib import Path
import numpy as np
from scipy.io import wavfile
import librosa
from tqdm import tqdm_notebook as tqdm


def read_file(filename, path='', sample_rate=None):
    ''' Reads in a wav file and returns it as an np.float32 array in the range [-1,1] '''
    filename = Path(path) / filename
    file_sr, data = wavfile.read(filename)
    if data.dtype == np.int16:
        data = np.float32(data) / np.iinfo(np.int16).max
    elif data.dtype != np.float32:
        raise OSError('Encounted unexpected dtype: {}'.format(data.dtype))
    if sample_rate is not None and sample_rate != file_sr:
        if len(data) > 0:
            data = librosa.core.resample(data, file_sr, sample_rate, res_type='kaiser_fast')
        file_sr = sample_rate
    return data, file_sr


def write_file(data, filename, path='', sample_rate=44100):
    ''' Writes a wav file to disk stored as int16 '''
    filename = Path(path) / filename
    if data.dtype == np.int16:
        int_data = data
    elif data.dtype == np.float32:
        int_data = np.int16(data * np.iinfo(np.int16).max)
    else:
        raise OSError('Input datatype {} not supported, use np.float32'.format(data.dtype))
    wavfile.write(filename, sample_rate, int_data)


def load_audio_files(path, filenames=None, sample_rate=None, trimmed=False):
    '''
    Loads in audio files and resamples if necessary.
    
    Args:
        path (str or PosixPath): directory where the audio files are located
        filenames (list of str): list of filenames to load. if not provided, load all 
                                 files in path
        sampling_rate (int): if provided, audio will be resampled to this rate
    
    Returns:
        list of audio files as numpy arrays, dtype np.float32 between [-1, 1]
    '''
    path = Path(path)
    if filenames is None:
        filenames = sorted(list(f.name for f in path.iterdir()))
    files = []
    for filename in tqdm(filenames, unit='files'):
        data, file_sr = read_file(filename, path, sample_rate=sample_rate)
        if trimmed:
            data = librosa.effects.trim(data)[0]
        files.append(data)
    return files
    
    
def resample_path(src_path, dst_path, sample_rate=16000):
    ''' Resamples a folder of wav files into a new folder at the given sample_rate '''
    src_path, dst_path = Path(src_path), Path(dst_path)
    dst_path.mkdir(exist_ok=True)
    filenames = list(src_path.iterdir())
    for filename in tqdm(filenames, unit="files"):
        data, file_sr = read_file(filename, sample_rate=sample_rate)
        write_file(data, dst_path/filename.name, sample_rate=sample_rate)
        

