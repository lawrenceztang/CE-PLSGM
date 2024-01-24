import torch.nn as nn
import torch.nn.functional as F
import torch


class Linear(nn.Module):
    def __init__(self, n_classes, n_dims):
        super().__init__()
        self.fc = nn.Linear(n_dims, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FC(nn.Module):
    def __init__(self, n_classes, n_dims):
        super().__init__()
        n_units = self._get_n_units()
        self.fc1 = nn.Linear(n_dims, n_units)
        scale = 1/(n_dims)**0.5
        torch.nn.init.uniform_(self.fc1.weight, a=-scale, b=scale)
        self.fc2 = nn.Linear(n_units, n_classes)
        scale = 1/n_units ** 0.5
        torch.nn.init.uniform_(self.fc2.weight, a=-scale, b=scale)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.softplus(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_n_units(self):
        raise NotImplementedError

class FC_10(FC):
    def _get_n_units(self):
        return 10
