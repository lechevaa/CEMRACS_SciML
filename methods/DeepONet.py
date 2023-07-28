import torch


class DeepONet(torch.nn.Module):
    def __init__(self, branch, trunk):
        super(DeepONet, self).__init__()
        self.branch = branch
        self.trunk = trunk

    def forward(self, u, y):
        weights = self.branch(u)
        basis = self.trunk(y)
        output = torch.matmul(weights, basis.T)
        return output

    def apply_method(self, u, y):
        return self.forward(u, y)
