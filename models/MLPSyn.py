import torch.nn as nn
import torch.nn.functional as nfunc


class MLPSyn(nn.Module):
    def __init__(self, out=2):
        super(MLPSyn, self).__init__()
        self.l1 = nn.Linear(100, 100)
        self.l2 = nn.Linear(100, out)

    def forward(self, h, update_batch_stats=True):
        h = self.l1(h)
        h = nfunc.relu(h)
        h = self.l2(h)
        logits = h
        return logits


class MLPSyn3(nn.Module):
    def __init__(self, out=2):
        super(MLPSyn3, self).__init__()
        self.l1 = nn.Linear(100, 100)
        self.l2 = nn.Linear(100, 10)
        self.l3 = nn.Linear(10, out)

    def forward(self, h, update_batch_stats=True):
        h = self.l1(h)
        h = nfunc.relu(h)
        h = self.l2(h)
        h = nfunc.relu(h)
        h = self.l3(h)
        logits = h
        return logits
