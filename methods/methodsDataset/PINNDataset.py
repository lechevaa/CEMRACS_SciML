from torch.utils.data import Dataset


class PINNDataset(Dataset):
    def __init__(self, x):

        self.x = x.float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
