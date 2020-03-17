import torch.nn as nn
import torch.nn.functional as nfunc


class CNN1l(nn.Module):
    """
    1 layer
    """
    def __init__(self, channel=3):
        super(CNN1l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 1, 3, 1, 0)
        self.conv1.bias.data.fill_(0)

    def forward(self, x):
        h = self.conv1(self.padding(x))
        return h


class CNN2l(nn.Module):
    """
    2 layers
    """
    def __init__(self, channel=3):
        super(CNN2l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 1, 3, 1, 0)
        self.conv2 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = self.conv2(self.padding(h))
        return h


class CNN2pool(nn.Module):
    """
    2 layers
    """
    def __init__(self, channel=3):
        super(CNN2pool, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 1, 3, 1, 0)
        self.conv2 = nn.Conv2d(1, 1, 3, 1, 0)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = self.pool1(h)
        h = self.conv2(self.padding(h))
        h = self.pool2(h)
        return h


class CNN3pool(nn.Module):
    """
    3 layers
    """
    def __init__(self, channel=3):
        super(CNN3pool, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 1, 3, 1, 0)
        self.conv2 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = self.pool1(h)
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = self.pool2(h)
        h = self.conv3(self.padding(h))
        h = self.pool3(h)
        return h


class CNN23l(nn.Module):
    """
    2 layers with 3 filters
    """
    def __init__(self, channel=3):
        super(CNN23l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 3, 3, 1, 0)
        self.conv2 = nn.Conv2d(3, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = self.conv2(self.padding(h))
        return h


class CNN3l(nn.Module):
    """
    3 layers
    """
    def __init__(self, channel=3):
        super(CNN3l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 1, 3, 1, 0)
        self.conv2 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = self.conv3(self.padding(h))
        return h


class CNN33l(nn.Module):
    """
    3 layers with 3 channels
    """
    def __init__(self, channel=3):
        super(CNN33l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 3, 3, 1, 0)
        self.conv2 = nn.Conv2d(3, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = self.conv3(self.padding(h))
        return h


class CNN43l(nn.Module):
    """
    4 layers
    """
    def __init__(self, channel=3):
        super(CNN43l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 3, 3, 1, 0)
        self.conv2 = nn.Conv2d(3, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv4 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = nfunc.relu(self.conv3(self.padding(h)))
        h = self.conv4(self.padding(h))
        return h


class CNN41l(nn.Module):
    """
    4 layers
    """
    def __init__(self, channel=3):
        super(CNN41l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 1, 3, 1, 0)
        self.conv2 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv4 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = nfunc.relu(self.conv3(self.padding(h)))
        h = self.conv4(self.padding(h))
        return h


class CNN45l(nn.Module):
    """
    4 layers
    """
    def __init__(self, channel=3):
        super(CNN45l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 5, 3, 1, 0)
        self.conv2 = nn.Conv2d(5, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv4 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = nfunc.relu(self.conv3(self.padding(h)))
        h = self.conv4(self.padding(h))
        return h


class CNN51l(nn.Module):
    """
    5 layers
    """
    def __init__(self, channel=3):
        super(CNN51l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 1, 3, 1, 0)
        self.conv2 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv4 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv5 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = nfunc.relu(self.conv3(self.padding(h)))
        h = nfunc.relu(self.conv4(self.padding(h)))
        h = self.conv5(self.padding(h))
        return h


class CNN53l(nn.Module):
    """
    5 layers
    """
    def __init__(self, channel=3):
        super(CNN53l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 3, 3, 1, 0)
        self.conv2 = nn.Conv2d(3, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv4 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv5 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = nfunc.relu(self.conv3(self.padding(h)))
        h = nfunc.relu(self.conv4(self.padding(h)))
        h = self.conv5(self.padding(h))
        return h


class CNN55l(nn.Module):
    """
    5 layers
    """
    def __init__(self, channel=3):
        super(CNN55l, self).__init__()
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(channel, 5, 3, 1, 0)
        self.conv2 = nn.Conv2d(5, 1, 3, 1, 0)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv4 = nn.Conv2d(1, 1, 3, 1, 0)
        self.conv5 = nn.Conv2d(1, 1, 3, 1, 0)

    def forward(self, x):
        h = nfunc.relu(self.conv1(self.padding(x)))
        h = nfunc.relu(self.conv2(self.padding(h)))
        h = nfunc.relu(self.conv3(self.padding(h)))
        h = nfunc.relu(self.conv4(self.padding(h)))
        h = self.conv5(self.padding(h))
        return h


class MLP1(nn.Module):
    def __init__(self, channel=0):
        """
        :param channel: just for consistent API, but not use here
        """
        super(MLP1, self).__init__()
        self.l1 = nn.Linear(100, 100)
        self.l2 = nn.Linear(100, 100)

    def forward(self, h):
        h = nfunc.relu(self.l1(h))
        h = self.l2(h)
        logits = h
        return logits
