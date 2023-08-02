from torch.utils.data import Dataset


class MLPINNDataset(Dataset):
    def __init__(self, x, y):

        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

