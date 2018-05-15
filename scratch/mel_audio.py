import librosa

# functions to produce melspectrogram in shape [1,1,128,128]; probably should be a class
# to use: pass a PosixPath to get_audio()


# a different way to adjust audio size
def adj_length(raw, length=65534): #length = ~1.5 seconds
    raw_max = np.argmax(raw)
    start = max(0, (raw_max-(length//2)))
    end = start+length
    if len(raw) < length:
        pad_width = (length-len(raw)//2)
        raw = np.pad(raw, (pad_width), 'constant')
    if (len(raw)-raw_max) < length:
        pad_width = (0, length-(len(raw)-raw_max))
        raw = np.pad(raw, pad_width, 'constant')
    return raw[start:end]

def open_audio(fn, sr=None):
    """Opens audio file using Librosa given the file path
    
    Arguments:
        fn: the file path for the audio
        
    Returns:
        The audio as a numpy array (TODO: of floats normalized to range between 0.0 - 1.0)
        and sampling rate
    """
    #flags = TODO
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            aud, sr = librosa.load(str(fn), sr=sr)#.astype(np.float32)
            if aud is None: raise OSError(f'File not recognized by librosa: {fn}')
            return aud, sr
        except Exception as e:
            raise OSError('Error handling audio at: {}'.format(fn)) from e


input_length=131070

# combination of weak-feature-extractor and kaggle starter kernel;
def get_mel(fn, sr=None, input_length=input_length, n_fft=1024, hop_length=512, n_mels=128):
    y, sr = open_audio(fn, sr)
   
 
    """
    # from kaggle starter kernel
    if len(y) > input_length:
            max_offset = len(y) - input_length
            offset = np.random.randint(max_offset)
            y = y[offset:(input_length+offset)]
    else:
        if input_length > len(y):
            max_offset = input_length - len(y)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        y = np.pad(y, (offset, input_length - len(y) - offset), "constant")
    """

    mel_feat = librosa.feature.melspectrogram(y,sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
    inpt = librosa.power_to_db(mel_feat).T

    #quick hack for now
    if inpt.shape[0] < 128:
        inpt = np.concatenate((inpt,np.zeros((128-inpt.shape[0],n_mels))),axis=0)

    # input needs to be 4D, batch_size X 1 X input_size[0] X input_size[1]
    inpt = np.reshape(inpt,(1,1,inpt.shape[0],inpt.shape[1]))
    return inpt

# returns melspectrogram in shape [1,1,128,128]
def get_audio(path, sr=None): return get_mel(str(path), sr)
