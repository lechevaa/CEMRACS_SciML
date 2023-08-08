import torch
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    def __init__(self, x, y, normalizers=None):
        if not normalizers:
            x_normalizer = None
            y_normalizer = UnitGaussianNormalizer(y)

            y = y_normalizer.encode(y)
        else:
            x_normalizer, y_normalizer = normalizers
            y = y_normalizer.encode(y)

        self.x = x.float()
        self.y = y.float()
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_normalizers(self):
        return self.x_normalizer, self.y_normalizer


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()