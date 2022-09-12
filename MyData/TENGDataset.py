from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
class TENGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, path, train=True):
        super(TENGDataset, self).__init__()
        dd = torch.load(path)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.label = dd['label_train']if train else dd['label_val']

    def __getitem__(self, index):
        x = self.data[index].astype('float32')
        target = self.label[index].astype('float32')
        # x = (x - np.min(x)) / (np.max(x) - np.min(x))
        # x = (x - np.mean(x)) / np.std(x)
        # x = transform(df, self.train)
        return x[None, :], target

    def __len__(self):
        return len(self.data)
