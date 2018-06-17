from fastai.imports import *
from fastai.layer_optimizer import *
from enum import IntEnum
from fastai.transforms import *
import librosa


#return raw audio at specified length
def adj_length(raw, length=10*44100): 
    raw_len = len(raw)
    if raw_len < length:
        raw = np.pad(raw, ((length-raw_len)//2), 'reflect')
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
hop_length=512
sr=44100

def get_mel(raw, sr=sr, feature_name='log_mel_spec', n_mels=n_mels, 
            n_fft=n_fft, hop_length=hop_length):

    if feature_name == 'mel_spec':
        feature = librosa.feature.melspectrogram(raw, sr=sr, fmin=125.0, fmax=7500.0, 
                                                 n_mels=n_mels, n_fft=n_fft, 
                                                 hop_length=hop_length)

    elif feature_name == 'log_mel_spec':
        mel_spec = librosa.feature.melspectrogram(raw, sr=sr, fmin=125.0, fmax=7500.0, 
                                                  n_mels=n_mels, n_fft=n_fft, 
                                                  hop_length=hop_length)
        log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=1.0, amin=1e-2)
        # amplitude_to_db normalizes to [-80.0, 0.0]
        feature = (log_mel_spec + 40.0) / 40.0  # rescale to [-1.0, 1.0]

    elif feature_name == 'mfcc':
        feature = librosa.feature.mfcc(raw, sr=sr, fmin=125.0, fmax=7500.0, 
                                       n_mfcc=n_mels, n_fft=n_fft, 
                                       hop_length=hop_length)
    else:
        raise NotImplementedError
    
    return feature.T


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


class Denormalize():
    """ De-normalizes an image, returning it to original format.
    """
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m


class Normalize():
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original image """
    def __init__(self, m, s, tfm_y=TfmType.NO):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
        self.tfm_y=tfm_y

    def __call__(self, x, y=None):
        x = (x-self.m)/self.s
        if self.tfm_y==TfmType.PIXEL and y is not None: y = (y-self.m)/self.s
        return x,y



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


def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None, tfm_y=None, sz_y=None, scale=None):
              # TODO: pad_mode=cv2.BORDER_REFLECT, 
    """
    Generate a standard set of transformations

    Arguments
    ---------
     normalizer :
         image normalizing function
     denorm :
         image denormalizing function
     sz :
         size, sz_y = sz if not specified.
     tfms :
         iterable collection of transformation functions
     max_zoom : float,
         maximum zoom
     pad : int,
         padding on top, left, right and bottom
     crop_type :
         crop type
     tfm_y :
         y axis specific transformations
     sz_y :
         y size, height
     pad_mode :
         cv2 padding style: repeat, reflect, etc.

    Returns
    -------
     type : ``Transforms``
         transformer for specified image operations.

    See Also
    --------
     Transforms: the transformer object returned by this function
    """
    pdb.set_trace()
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    if sz_y is None: sz_y = sz
    if scale is None:
        pass
        #TODO
        #scale = [RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y) if max_zoom is not None
         #        else Scale(sz, tfm_y, sz_y=sz_y)]
    elif not is_listy(scale): pass #TODO: scale = [scale]
    #TODO: if pad: scale.append(AddPadding(pad, mode=pad_mode))
    # Take out ?? - if crop_type!=CropType.GOOGLENET: tfms=scale+tfms
    return Transforms(sz, tfms, normalizer, denorm, crop_type, tfm_y=tfm_y, sz_y=sz_y)

def noop(x):
    """dummy function for do-nothing.
    equivalent to: lambda x: x"""
    return x

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

