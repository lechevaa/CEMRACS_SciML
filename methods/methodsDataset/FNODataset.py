import torch
from torch.utils.data import Dataset


class FNODataset(Dataset):
    def __init__(self, x, y):
        nx = y.shape[1]
        x = torch.Tensor(x).view(-1, 1)
        x = x.repeat(1, nx).unsqueeze(-1)
        y = torch.Tensor(y)
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

