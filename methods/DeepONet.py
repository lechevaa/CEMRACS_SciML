import torch


class DeepONet(torch.nn.Module):
    def __init__(self, branch, trunk):
        super(DeepONet, self).__init__()
        self.branch = branch
        self.trunk = trunk

    def forward(self, u_, y_):
        weights = self.branch(u_)
        basis = self.trunk(y_)
        output = torch.matmul(weights, basis.T)
        return output
