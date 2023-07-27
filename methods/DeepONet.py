import torch


class DeepONet(torch.nn.Module):
    def __init__(self, branch, trunk):
        super(DeepONet, self).__init__()
        self.branch = branch
        self.trunk = trunk

    def forward(self, u, y):
        print('u ', u.shape, 'y ', y.shape)
        weights = self.branch(u)
        basis = self.trunk(y)
        print('trunk ', basis.shape, 'branch ', weights.shape)
        output = torch.matmul(weights, basis)
        return output

    def apply_method(self, u, y):
        return self.forward(u, y)
