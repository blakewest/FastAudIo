import librosa
from librosa import display
import csv

from fastai.imports import *
from fastai.torch_imports import *
from fastai.core import *
from fastai.transforms import *
from fastai.layer_optimizer import *
from fastai.dataloader import DataLoader
from fastai.dataset import *
#from audio_transforms import *



# functions to produce melspectrogram in shape [1,1,128,128]; probably should be a class
# I adjusted the n-mels to 256, but left the hop_length and n_fft the same; not sure how
# that affects the quality of the spectrogram


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

def open_audio(fn, length=3*44100, sr=None):
    """Opens raw audio file using Librosa given the file path
    
    Arguments:
        fn: the file path for the audio
        sr: sample-rate (None maintains sr of original)
        
    Returns:
        aud: audio as a numpy array (TODO: of floats normalized to range between 0.0 - 1.0)
        sr: sampling rate
        l: length of aud array
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
            aud = adj_length(aud, length)
            aud = np.reshape(aud, (1, aud.shape[0]))
            return aud#, sr#, l
        except Exception as e:
            raise OSError('Error handling audio at: {}'.format(fn)) from e

"""
# replacing with John's process_audio_utils version
# combination of weak-feature-extractor and kaggle starter kernel;
def get_mel(y, sr=44100, n_fft=1024, hop_length=512, n_mels=128): # add input_length param ??
    #y = open_audio(fn)
    #y, sr = open_audio(fn, sr)
    #y = adj_length(y)
    
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
    
    mel_feat = librosa.feature.melspectrogram(y[0,:],sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
    inpt = librosa.power_to_db(mel_feat).T

    #quick hack for now
    if inpt.shape[0] < 128:
        inpt = np.concatenate((inpt,np.zeros((128-inpt.shape[0],n_mels))),axis=0)

    # input needs to be 4D, batch_size X 1 X input_size[0] X input_size[1]
    inpt = np.reshape(inpt,(1,inpt.shape[0],inpt.shape[1]))
    return inpt
"""
# returns raw or aelspectrogram if melspect=True
def get_audio(path, melspect=False): 
    if melspect: 
        return get_mel(open_audio(path))
    else:
        return open_audio(path)


class PassthruDataset(Dataset):
    def __init__(self,*args, is_reg=True, is_multi=False):
        *xs,y=args
        self.xs,self.y = xs,y
        self.is_reg = is_reg
        self.is_multi = is_multi

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return [o[idx] for o in self.xs] + [self.y[idx]]

    @classmethod
    def from_data_frame(cls, df, cols_x, col_y, is_reg=True, is_multi=False):
        cols = [df[o] for o in cols_x+[col_y]]
        return cls(*cols, is_reg=is_reg, is_multi=is_multi)

class BaseDataset(Dataset):
    """An abstract class representing a fastai dataset, it extends torch.utils.data.Dataset."""
    def __init__(self, transform=None):
        self.transform = transform
        self.n = self.get_n()
        self.c = self.get_c()
        self.sz = self.get_sz()

    def get1item(self, idx):
        x,y = self.get_x(idx),self.get_y(idx)
        #x = adj_length(x)
        return self.get(self.transform, x, y)

    def __getitem__(self, idx):
        if isinstance(idx,slice):
            xs,ys = zip(*[self.get1item(i) for i in range(*idx.indices(self.n))])
            return np.stack(xs),ys
        return self.get1item(idx)

    def __len__(self): return self.n

    def get(self, tfm, x, y):
        return (x,y) if tfm is None else tfm(x,y)

    @abstractmethod
    def get_n(self):
        """Return number of elements in the dataset == len(self)."""
        raise NotImplementedError

    @abstractmethod
    def get_c(self):
        """Return number of classes in a dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_sz(self):
        """Return maximum size of an image in a dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_x(self, i):
        """Return i-th example (image, wav, etc)."""
        raise NotImplementedError

    @abstractmethod
    def get_y(self, i):
        """Return i-th label."""
        raise NotImplementedError

    @property
    def is_multi(self):
        """Returns true if this data set contains multiple labels per sample."""
        return False

    @property
    def is_reg(self):
        """True if the data set is used to train regression models."""
        return False


class FilesDataset(BaseDataset):
    def __init__(self, fnames, transform, path):
        self.path,self.fnames = path,fnames
        super().__init__(transform)
    def get_sz(self): return 128 #return self.transform.sz
    def get_x(self, i): return open_audio(os.path.join(self.path, self.fnames[i]), sr=None)
        #return get_audio(os.path.join(self.path, self.fnames[i]), sr=None) 
        # from Image Dataset
        # return open_image(os.path.join(self.path, self.fnames[i]))
    def get_n(self): return len(self.fnames)

    def resize_imgs(self, targ, new_path):
        dest = resize_imgs(self.fnames, targ, self.path, new_path)
        return self.__class__(self.fnames, self.y, self.transform, dest)

    def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))

class FilesArrayDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return self.y[i]
    def get_c(self):
        return self.y.shape[1] if len(self.y.shape)>1 else 0

class FilesIndexArrayDataset(FilesArrayDataset):
    def get_c(self): return int(self.y.max())+1


class FilesNhotArrayDataset(FilesArrayDataset):
    @property
    def is_multi(self): return True


class FilesIndexArrayRegressionDataset(FilesArrayDataset):
    def is_reg(self): return True

class ArraysDataset(BaseDataset):
    def __init__(self, x, y, transform):
        self.x,self.y=x,y
        assert(len(x)==len(y))
        super().__init__(transform)
    def get_x(self, i): return self.x[i]
    def get_y(self, i): return self.y[i]
    def get_n(self): return len(self.y)
    def get_sz(self): return self.x.shape[1]


class ArraysIndexDataset(ArraysDataset):
    def get_c(self): return int(self.y.max())+1
    def get_y(self, i): return self.y[i]


class ArraysNhotDataset(ArraysDataset):
    def get_c(self): return self.y.shape[1]
    @property
    def is_multi(self): return True
    
class ModelData():
    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path,self.trn_dl,self.val_dl,self.test_dl = path,trn_dl,val_dl,test_dl

    @classmethod
    def from_dls(cls, path,trn_dl,val_dl,test_dl=None):
        #trn_dl,val_dl = DataLoader(trn_dl),DataLoader(val_dl)
        #if test_dl: test_dl = DataLoader(test_dl)
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg
    @property
    def is_multi(self): return self.trn_ds.is_multi
    @property
    def trn_ds(self): return self.trn_dl.dataset
    @property
    def val_ds(self): return self.val_dl.dataset
    @property
    def test_ds(self): return self.test_dl.dataset
    @property
    def trn_y(self): return self.trn_ds.y
    @property
    def val_y(self): return self.val_ds.y


class AudioData(ModelData):
    def __init__(self, path, datasets, bs, num_workers, classes):
        trn_ds,val_ds,fix_ds,aug_ds,test_ds,test_aug_ds = datasets
        self.path,self.bs,self.num_workers,self.classes = path,bs,num_workers,classes
        self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl,self.test_dl,self.test_aug_dl = [
            self.get_dl(ds,shuf) for ds,shuf in [
                (trn_ds,True),(val_ds,False),(fix_ds,False),(aug_ds,False),
                (test_ds,False),(test_aug_ds,False)
            ]
        ]

    def get_dl(self, ds, shuffle):
        if ds is None: return None
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=False)
    
    @property
    def sz(self): return self.trn_ds.sz
    @property
    def c(self): return self.trn_ds.c

    def resized(self, dl, targ, new_path):
        return dl.dataset.resize_imgs(targ,new_path) if dl else None

    def resize(self, targ_sz, new_path='tmp'):
        new_ds = []
        dls = [self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl]
        if self.test_dl: dls += [self.test_dl, self.test_aug_dl]
        else: dls += [None,None]
        t = tqdm_notebook(dls)
        for dl in t: new_ds.append(self.resized(dl, targ_sz, new_path))
        t.close()
        return self.__class__(new_ds[0].path, new_ds, self.bs, self.num_workers, self.classes)

    @staticmethod
    def get_ds(fn, trn, val, tfms, test=None, **kwargs):
        res = [
            fn(trn[0], trn[1], tfms[0], **kwargs), # train
            fn(val[0], val[1], tfms[1], **kwargs), # val
            fn(trn[0], trn[1], tfms[1], **kwargs), # fix
            fn(val[0], val[1], tfms[0], **kwargs)  # aug
        ]
        if test is not None:
            if isinstance(test, tuple):
                test_lbls = test[1]
                test = test[0]
            else:
                test_lbls = np.zeros((len(test),1))
            res += [
                fn(test, test_lbls, tfms[1], **kwargs), # test
                fn(test, test_lbls, tfms[0], **kwargs)  # test_aug
            ]
        else: res += [None,None]
        return res


class AudioClassifierData(AudioData):
    @classmethod
    def from_names_and_array(cls, path, fnames,y,classes, val_idxs=None, test_name=None,
            num_workers=8, suffix='', tfms=(None,None), bs=64, continuous=False):
        val_idxs = get_cv_idxs(len(fnames)) if val_idxs is None else val_idxs
        ((val_fnames,trn_fnames),(val_y,trn_y)) = split_by_idx(val_idxs, np.array(fnames), y)

        test_fnames = read_dir(path, test_name) if test_name else None
        if continuous: f = FilesIndexArrayRegressionDataset
        else:
            f = FilesIndexArrayDataset if len(trn_y.shape)==1 else FilesNhotArrayDataset
        datasets = cls.get_ds(f, (trn_fnames,trn_y), (val_fnames,val_y), tfms,
                               path=path, test=test_fnames)
        return cls(path, datasets, bs, num_workers, classes=classes) 
