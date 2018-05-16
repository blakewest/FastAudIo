class BasicAudioModel(nn.Module):
    def __init__(self, num_classes, bs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, 9),
            nn.ReLU(),
            nn.Conv1d(16, 16, 9),
            nn.ReLU(),
            nn.MaxPool1d(16),
            nn.Dropout(0.1),
            nn.Conv1d(16, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Dropout(0.01),
            nn.Conv1d(32, 256, 3),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3),
            nn.Dropout(0.01),
            nn.MaxPool1d(4),
            nn.Conv1d(256, num_classes, 3),
            Lambda(lambda tensor: torch.mean(tensor, 2, keepdim=True)),
            Lambda(lambda tensor: tensor.view(tensor.shape[0], -1)),
        )

    def forward(self, input):
        return self.model(input)

def get_trn_val_split(x, y, val_pct=0.15):
    val_idxs = get_cv_idxs(len(x), val_pct=val_pct)
    if isinstance(x, list):
        return [([arr[i] for i in val_idxs], [arr[i] for i in range(len(arr)) if i not in val_idxs]) for arr in [x,y]]
    else:
        return split_by_idx(val_idxs, x, y)

class AudioLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data):
        return F.cross_entropy

def load_audio_from_df(trn_path, trn_df, sample_rate=16000):
    return [retrieve_file(str(trn_path) + '/' + trn_df['fname'][i], sample_rate=sample_rate) for i in range(len(trn_df))]

def retrieve_file(filepath, sample_rate=16000):
    data, _ = librosa.core.load(filepath, sr=sample_rate, res_type='kaiser_fast')
    return data

def preprocess_audio(audio_files):
    norm = librosa.util.normalize
    trimmed_xs = [librosa.effects.trim(norm(x))[0] for x in audio_files]
    return [x.reshape(1, x.shape[0]) for x in trimmed_xs]

def preprocess_ys(labels, one_hot=False):
    if isinstance(labels[0], str):
        tok2int = {v:k for k,v in enumerate(np.unique(labels))}
        labels = np.array([tok2int[tok] for tok in labels])
    num_classes = len(np.unique(labels))
    if one_hot:
        return [one_hot(labels[i], num_classes).reshape(1, num_classes) for i in range(len(labels))]
    else:
        return labels

class AudioModelData():
    def __init__(self, path, trn_ds, val_ds, test_ds=None, bs=64, sample_rate=16000):
        self.path = path
        self.bs = bs
        self.trn_ds, self.val_ds, self.test_ds = trn_ds, val_ds, test_ds
        self.trn_dl = AudioDataLoader(trn_ds, bs, sampler=SortishSampler(trn_ds, key=lambda x: len(trn_ds[x][0][0]), bs=bs))
        self.val_dl = AudioDataLoader(val_ds, bs, sampler=SortishSampler(val_ds, key=lambda x: len(val_ds[x][0][0]), bs=bs))
        self.test_dl = AudioDataLoader(test_ds, bs, sampler=SortishSampler(test_ds, key=lambda x: len(test_ds[x][0][0]),bs=bs)) if test_ds is not None else None
        self.num_classes = self.trn_dl.dataset.get_c()

    @classmethod
    def from_path_and_dataframes(cls, trn_path, trn_df, test_path=None, test_df=None, val_path=None, val_df=None, val_pct=0.15, model_path='./', bs=64):
        xs = load_audio_from_df(trn_path, trn_df)
        xs = preprocess_audio(x)
        ys = preprocess_ys(trn_df['label'])
        if test_path is not None:
            text_xs = load_audio_from_df(test_path, test_df)
            text_xs = preprocess_audio(test_xs)
        else:
            test_x = None
        return cls.from_array(xs, ys, test_xs, bs=bs)

    @classmethod
    def from_array(cls, trn_x, trn_y, test_x=None, val_pct=0.15, bs=64, model_path="./", **kwargs):
        ((val_x, trn_x), (val_y, trn_y)) = get_trn_val_split(trn_x, trn_y, val_pct)
        trn_ds = AudioDataset(trn_x, trn_y)
        val_ds = AudioDataset(val_x, val_y)
        test_ds = AudioDataset(test_x, test_y) if test_x is not None else None
        return cls(model_path, trn_ds, val_ds, test_ds, bs=bs)

    def get_model(self, optimizer=torch.optim.Adam):
        basic_model = BasicAudioModel(self.num_classes, self.bs)
        model = SingleModel(to_gpu(basic_model))
        return AudioLearner(self, model, opt_fn=optimizer)

class AudioDataLoader(DataLoader):
    def get_batch(self, indexes):
        batch_data = [self.dataset[i] for i in indexes]
        x_lens = [len(item[0][0]) for item in batch_data]
        if len(np.unique(x_lens)) > 1:
            max_len = np.max(x_lens)
            for i, item in enumerate(batch_data):
                item = list(item)
                clip_len = len(item[0][0])
                item[0] = np.pad(item[0], ((0,0), (0, max_len-clip_len)), 'wrap')
                batch_data[i] = tuple(item)
        return self.np_collate(batch_data)

class AudioDataset(BaseDataset):
    def __init__(self, xs, ys, transforms=None):
        if isinstance(ys[0], str):
            ys = preprocess_ys(ys)
        self.ys = ys
        self.xs = xs
        assert(len(xs) == len(xs)), "Length of xs does not equal length of ys"
        super().__init__(transforms)

    def get_x(self, i):
        return self.xs[i]

    def get_y(self, i):
        return self.ys[i]

    def get_n(self):
        return len(self.xs)

    def get_sz(self):
        return self.get_x(1).shape[0]

    def get_c(self):
        return int(np.max(self.ys) + 1)

def create_sample_data(path, percent=0.1, sample_path=None, overwrite=False, labels_df=None):
    sample_path = path + '_sample' if sample_path is None else sample_path
    if not os.path.exists(sample_path):
            print("Creating folder for the sample set...", sample_path)
            os.mkdir(sample_path)
    existing_sample_files = glob(sample_path + '/*')
    if len(existing_sample_files) > 0:
        if not overwrite:
            print("Sample already exists. Pass overwrite=True to delete and redo")
            return
        else:
            for file in existing_sample_files:
                os.remove(file)
    print("Saving a", percent * 100, "percent sample to", sample_path)
    for filepath in glob(path + '/*'):
        if np.random.random() < percent:
            fname = filepath.split('/')[-1]
            copyfile(filepath, sample_path + '/' + fname)
    if labels_df:
        sample_fnames = [filepath.split('/')[-1] for filepath in glob(sample_path + '/*')]
        labels_df[labels_df['fname'].isin(sample_fnames)].to_csv(path + '../train_sample.csv')

def save_data(data, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

def load_data(filename):
    with open(filename, 'rb') as infile:
        result = pickle.load(infile)
    return result

def display_sample(train, category=None):
    sample = train[train['label'] == category].sample() if category else train.sample()
    fname = str(TRN_PATH/sample['fname'].values[0])
    print(sample)
    return ipd.Audio(fname)

def munge_and_save_data(trn_path, trn_df, x_filepath, y_filepath):
    print("Loading files...")
    xs = load_audio_from_df(trn_path, trn_df)
    print("Processing audio...")
    xs = preprocess_audio(xs)
    print("Processing labels...")
    ys = preprocess_ys(trn_df['label'])
    print("Saving xs and ys...")
    save_data(xs, x_filepath)
    save_data(ys, y_filepath)
