from fastai.imports import *
from fastai.layer_optimizer import *
from enum import IntEnum
from fastai.transforms import *
import librosa


#return raw audio at specified length
def adj_length(raw, length): 
    raw_len = len(raw)
    if raw_len < length:
        raw = np.pad(raw, ((length-raw_len)//2), 'wrap')
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

n_mels=128
n_fft=1024  
hop_length=256
#sr=44100

def get_mel(raw, sr, feature_name='log_mel_spec', n_mels=n_mels, 
            n_fft=n_fft, hop_length=hop_length):

    if feature_name == 'stft':
        D = librosa.core.stft(raw, n_fft=n_fft, hop_length=hop_length)
        mag, _ = librosa.core.magphase(D, power=2)
        feature = (librosa.power_to_db(mag, ref=1.0)+40.0) / 40.0

    elif feature_name == 'log_mel_spec':
        mel_spec = librosa.feature.melspectrogram(raw, sr=sr, 
                                                  n_mels=n_mels, n_fft=n_fft, 
                                                  hop_length=hop_length)
        log_mel_spec = librosa.amplitude_to_db(mel_spec, amin=1e-2)
        # amplitude_to_db normalizes to [-80.0, 0.0]
        feature = (log_mel_spec + 40.0) / 40.0  # rescale to [-1.0, 1.0]

    elif feature_name == 'mfcc':
        feature = librosa.feature.mfcc(raw, sr=sr, fmin=125.0, fmax=7500.0, 
                                       n_mfcc=n_mels, n_fft=n_fft, 
                                       hop_length=hop_length)
    else:
        raise NotImplementedError
    
    return feature

'''
n_mels=128
hop_length=256
n_fft=[1024, 2048, 4096]
sr=44100

def get_mel(raw, sr=sr, feature_name='log_mel_spec', n_mels=n_mels, 
            hop_length=hop_length):

    feature = []
    for i in range(len(n_fft)):
        mel_spec = librosa.feature.melspectrogram(raw, sr=sr,
                                                  n_mels=n_mels, n_fft=n_fft[i], 
                                                  hop_length=hop_length)
        log_mel_spec = librosa.amplitude_to_db(mel_spec, amin=1e-2)
        # amplitude_to_db normalizes to [-80.0, 0.0]
        feature.append((log_mel_spec + 40.0) / 40.0)  # rescale to [-1.0, 1.0]
    features = np.stack((feature[0],feature[1],feature[2]), axis=2)
    #features = np.array(feature)
    return features
'''

class TfmType(IntEnum):
    """ Type of transformation.
    Parameters
        IntEnum: predefined types of transformations
            NO:    the default, y does not get transformed when x is transformed.
            PIXEL: x and y are images and should be transformed in the same way.
                   Example: image segmentation.
            COORD: y are coordinates (i.e bounding boxes)
            CLASS: y are class labels (same behaviour as PIXEL, except no normalization)
    """
    NO = 1
    PIXEL = 2
    COORD = 3
    CLASS = 4

class Transforms():
    def __init__(self, sz, tfms, normalizer, denorm, tfm_y=TfmType.NO, sz_y=None): #crop_type=CropType.CENTER, tfm_y=TfmType.NO, 

        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        #crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms
        #self.tfms.append(crop_tfm)
        if normalizer is not None: self.tfms.append(normalizer)
        #self.tfms.append(ChannelOrder(tfm_y))

    def __call__(self, im, y=None): return compose(im, y, self.tfms)
    def __repr__(self): return str(self.tfms)


transforms_basic    = [RandomRotate(10), RandomLighting(0.05, 0.05)]
transforms_side_on  = transforms_basic + [RandomFlip()]
transforms_top_down = transforms_basic + [RandomDihedral()]

#aud_transform_basic = [get_mel(feature_name='mel_spec'), get_mel(feature_name='mfcc')

imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
"""Statistics pertaining to image data from image net. mean and std of the images of each color channel"""
inception_stats = A([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
inception_models = (inception_4, inceptionresnet_2)

def tfms_from_stats(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=None, tfm_y=None, sz_y=None, norm_y=True, scale=None): #TODO: crop_type=CropType.RANDOM, pad_mode=cv2.BORDER_REFLECT, 
		    
    """ Given the statistics of the training image sets, returns separate training and validation transform functions
    """
    if aug_tfms is None: aug_tfms=[]
    tfm_norm = Normalize(*stats, tfm_y=tfm_y if norm_y else TfmType.NO) if stats is not None else None
    tfm_denorm = Denormalize(*stats) if stats is not None else None
    val_crop = None #CropType.CENTER if crop_type in (CropType.RANDOM,CropType.GOOGLENET) else crop_type
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=val_crop, tfm_y=tfm_y, 
                        sz_y=sz_y, scale=scale)
    trn_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=crop_type, tfm_y=tfm_y, 
                        sz_y=sz_y, tfms=aug_tfms, max_zoom=max_zoom, pad_mode=pad_mode, 
                        scale=scale)
    return trn_tfm, val_tfm


def tfms_from_model(f_model, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM, tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, norm_y=True, scale=None):
    """ Returns separate transformers of images for training and validation.
    Transformers are constructed according to the image statistics given b y the model. (See tfms_from_stats)

    Arguments:
        f_model: model, pretrained or not pretrained
    """
    stats = inception_stats if f_model in inception_models else imagenet_stats
    return tfms_from_stats(stats, sz, aug_tfms, max_zoom=max_zoom, pad=pad, crop_type=crop_type, tfm_y=tfm_y, sz_y=sz_y, pad_mode=pad_mode, norm_y=norm_y, scale=scale)


class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """
    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.store = threading.local()

    def set_state(self): pass
    def __call__(self, x, y):
        self.set_state()
        x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y in (TfmType.PIXEL, TfmType.CLASS)
                else self.transform_coord(x,y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x),y

    def transform(self, x, y=None):
        x = self.do_transform(x,False)
        return (x, self.do_transform(y,True)) if y is not None else x

    @abstractmethod
    def do_transform(self, x, is_y): raise NotImplementedError


class Denormalize():
    """ De-normalizes an image, returning it to original format.
    """
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m


class Normalize():
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original image """
    def __init__(self, m, s): #tfm_y=TfmType.NO
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
        #self.tfm_y=tfm_y

    def __call__(self, x, y=None):
        x = (x-self.m)/self.s
        #if self.tfm_y==TfmType.PIXEL and y is not None: y = (y-self.m)/self.s
        return x,y

class ChannelOrder():
    '''
    changes image array shape from (h, w, 3) to (3, h, w). 
    tfm_y decides the transformation done to the y element. 
    '''
    def __init__(self, tfm_y=TfmType.NO): self.tfm_y=tfm_y

    def __call__(self, x, y):
        x = np.rollaxis(x, 2)
        #if isinstance(y,np.ndarray) and (len(y.shape)==3):
        #if self.tfm_y==TfmType.PIXEL: y = np.rollaxis(y, 2)
        #elif self.tfm_y==TfmType.CLASS: y = y[...,0]
        return x,y

def vocode(x,y,rate=2.0):
    return librosa.phase_vocoder(x, rate), y

def rand0(s): return random.random()*(s*2)-s

def rand1(s): return int(random.random()*s)

def focus_mel(aud, b, c):
    ''' highlights audio's mel_bands'''
    if b == 0: return aud
    mu = np.average(aud[:b])
    return aud[:b]+mu*c

class RandomFocus_mel(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c
        
    def set_state(self):
        self.store.b_rand = rand1(self.b)
        self.store.c_rand = self.c
        
    def do_transform(self, x, is_y):
        b = self.store.b_rand
        c = self.store.c_rand
        x = focus_mel(x, b, c)
        return x

def lighting(im, b, c):
    ''' adjusts image's balance and contrast'''
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        #if is_y and self.tfm_y != TfmType.PIXEL: return x
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x
       
def compose(im, y, fns):
    """ apply a collection of transformation functions fns to images
    """
    for fn in fns:
        #pdb.set_trace()
        im, y =fn(im, y)
    return im if y is None else (im, y)


class Transforms():
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms
        self.tfms.append(crop_tfm)
        if normalizer is not None: self.tfms.append(normalizer)
        self.tfms.append(ChannelOrder(tfm_y))

    def __call__(self, im, y=None): return compose(im, y, self.tfms)
    def __repr__(self): return str(self.tfms)


def noop(x):
    """dummy function for do-nothing.
    equivalent to: lambda x: x"""
    return x

class AudTransforms():
    def __init__(self, tfms, normalizer, denorm):
        #if sz_y is None: sz_y = sz
        #self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        self.denorm,self.norm = denorm,normalizer
        #pdb.set_trace()
        #crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms
        #self.tfms.append(crop_tfm)
        if normalizer is not None: self.tfms.append(normalizer)
        #self.tfms.append(ChannelOrder())

    def __call__(self, im, y=None): return compose(im, y, self.tfms) 
    def __repr__(self): return str(self.tfms)

def audio_gen(normalizer, denorm, tfms=None):
    if tfms is None: tfms = []
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    return AudTransforms(tfms, normalizer, denorm)

def aud_tfms_from_stats(stats, aug_tfms=None):
#def tfms_from_stats(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    #tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, norm_y=True, scale=None):
    """ Given the statistics of the training image sets, returns separate training and validation transform functions
    """
    if aug_tfms is None: aug_tfms=[]
    #tfm_norm = Normalize(*stats, tfm_y=tfm_y if norm_y else TfmType.NO) if stats is not None else None
    tfm_norm = Normalize(*stats) if stats is not None else None
    tfm_denorm = Denormalize(*stats) if stats is not None else None
    #val_crop = CropType.CENTER if crop_type in (CropType.RANDOM,CropType.GOOGLENET) else crop_type
    #val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=val_crop,
            #tfm_y=tfm_y, sz_y=sz_y, scale=scale)
    val_tfm = audio_gen(tfm_norm, tfm_denorm)
    #trn_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=crop_type,
            #tfm_y=tfm_y, sz_y=sz_y, tfms=aug_tfms, max_zoom=max_zoom, pad_mode=pad_mode, scale=scale)
    trn_tfm = audio_gen(tfm_norm, tfm_denorm, tfms=aug_tfms)
    return trn_tfm, val_tfm

