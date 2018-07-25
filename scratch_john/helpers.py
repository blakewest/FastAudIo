from pathlib import Path
import numpy as np
import torch.nn as nn
from fastai.dataset import get_cv_idxs, split_by_idx, BaseDataset, ModelData
from fastai.dataloader import DataLoader
from fastai.transforms import compose, lighting
import cv2

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    

# from blake
def get_trn_val_split(x, y, val_pct=0.15):
    val_idxs = get_cv_idxs(len(x), val_pct=val_pct)
    if isinstance(x, list):
        return [([arr[i] for i in val_idxs], 
                 [arr[i] for i in range(len(arr)) if i not in val_idxs]) 
                for arr in [x,y]]
    else:
        return split_by_idx(val_idxs, x, y)
    

def pad_to_longest(items, dim=0, pad_mode='wrap'):    
    assert len(items) > 0
    total_dim = len(items[0].shape)        
    assert dim <= total_dim - 1
    pad_template = ((0,0),) * total_dim
    longest_len = max(item.shape[dim] for item in items)
    
    for i in range(len(items)):
        item = items[i]
        item_len = item.shape[dim]
        pad_len = longest_len - item_len
        if pad_len > 0:
            if item_len == 0:
                pad_mode = 'constant'  # cannot wrap an empty array
            pad_width = [(0,pad_len) if d == dim else (0,0) 
                        for d in range(total_dim)]
            items[i] = np.pad(item, pad_width, pad_mode)
    return items



def mu_law_encode(data, mu=256):
    magnitude = np.log(1.0 + mu*np.abs(data)) / np.log(1.0 + mu)
    return np.sign(data) * magnitude

def mu_law_decode(data, mu=256):
    magnitude = (np.exp(np.abs(data) * np.log(1.0 + mu)) - 1.0) / mu
    return np.sign(data) * magnitude

def quantize(data, n_bins=256):
    bins = np.linspace(-1, 1, n_bins)
    return np.digitize(data, bins) - 1


def one_hot(data, c=None):
    assert len(data.shape) == 1, 'Input may only have one dimension'
    if c is None:
        c = data.max() + 1
    output = np.zeros((data.size, c), dtype=np.float32)
    output[np.arange(data.size), data] = 1
    return output


def save_submission(predictions, labels, test_df, filename):
    # From the fizzbuzz starter kernel
    top_3 = np.array(labels)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test_df.label = predicted_labels
    test_df.to_csv(filename, index=False)
    
    
# TODO: these could definitely be refactored into one..

class AudioDataLoader1d(DataLoader):
    def get_batch(self, indexes):
        batch_data = [self.dataset[i] for i in indexes]
        xs = [x for x,y in batch_data]
        ys = [y for x,y in batch_data]
        xs = pad_to_longest(xs, dim=0)
        return self.np_collate(list(zip(xs, ys)))
    
class AudioDataLoader2d(DataLoader):
    def get_batch(self, indexes):
        batch_data = [self.dataset[i] for i in indexes]
        xs = [x for x,y in batch_data]
        ys = [y for x,y in batch_data]
        xs = pad_to_longest(xs, dim=1)
        return self.np_collate(list(zip(xs, ys)))
    
    
class AudioDataset(BaseDataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        assert len(x) == len(y)
        super().__init__(transform)
    def get_x(self, i):
        return self.x[i]
    def get_y(self, i):
        return self.y[i]
    def get_n(self):
        return len(self.y)
    def get_c(self):
        return max(self.y) + 1
    def get_sz(self):
        return 0 # doesn't make sense

    
class AudioFilesDataset(BaseDataset):
    def __init__(self, path, fnames, y, use_tfms=False, transform=None):
        self.path = Path(path)
        self.fnames = fnames
        self.y = y
        self.use_tfms = use_tfms
        assert len(fnames) == len(y)
        super().__init__(transform)
    def get_x(self, i):
        fname = self.fnames[i]
        if self.use_tfms:
            fname = f'{fname[:-4]}_{np.random.randint(10)}.wav'
        fname = self.path/f'{fname}.npy'
        return np.load(fname)
    def get_y(self, i):
        return self.y[i]
    def get_n(self):
        return len(self.y)
    def get_c(self):
        return max(self.y) + 1
    def get_sz(self):
        return self.get_x(0).shape[0]
    


# Functions to split the training data into evenly balanced folds

def split_into_folds(arr, n_folds):
    np.random.seed(0)
    labels, counts = np.unique(arr, return_counts=True)
    folds = [[] for _ in range(n_folds)]
    for (label, count) in zip(labels, counts):
        idx = np.where(arr == label)[0]
        splits = np.array_split(idx, n_folds)
        for fold, split in zip(folds, splits):
            fold.append(split)
    folds_np = []
    for fold in folds:
        fold_np = np.concatenate(fold)
        np.random.shuffle(fold_np)
        folds_np.append(fold_np)
    return folds_np


def get_val_idx(arr, n_folds):
    folds = split_into_folds(arr, n_folds)
    for n in range(n_folds):
        yield folds[n]
        
        
def get_trn_val_split_from_folds(x, y, val_idx):
    return [([arr[i] for i in val_idx], 
             [arr[i] for i in range(len(arr)) if i not in val_idx]) 
            for arr in [x,y]]


def get_first_split(x, y, n_folds=8):
    val_idx = next(get_val_idx(y, n_folds))
    return get_trn_val_split_from_folds(x, y, val_idx)



# Transform Functions

class Transforms():
    def __init__(self, tfms):
        self.tfms = tfms
    def __call__(self, x, y=None): 
        return compose(x, y, self.tfms)
    def __repr__(self):
        return str(self.tfms)
    
    
# NOTE: resizing the x (time) dimension did not seem to help at all
class RandomPitchTimeShift():
    def __init__(self, min_x=0.8, max_x=1.2, max_y=1.2):
        self.min_x = min_x
        self.max_x = max_x
        self.max_y = max_y
    def __call__(self, x, y=None):
        fx = np.random.uniform(self.min_x, self.max_x)
        fy = np.random.uniform(1.0, self.max_y)
        x = cv2.resize(x, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)[:x.shape[0],:]
        return x, y
    
    
class RandomLight():
    def __init__(self, a=0.1, b=0.1):
        self.a = a
        self.b = b
    def __call__(self, x, y=None):
        a = np.random.uniform(-self.a, self.a)
        b = np.random.uniform(-self.b, self.b)
        b = -1/(b-1) if b<0 else b+1
        
        xmin, xmax = x.min(), x.max()
        diff = xmax - xmin
        x = (x-xmin) / diff
        x = lighting(x, a, b)
        x = (x * diff) + xmin
        return x, y
