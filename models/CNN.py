import torch.nn as nn
import torch.nn.functional as nfunc

from torch_func.utils import call_bn

EPS = 1e-5
MOMENTUM = 0.1


class CNN9(nn.Module):
    """
    Ref: [VAT Chainer](https://github.com/takerum/vat_chainer/blob/master/models/cnn.py)
    [VAT TF[(https://github.com/takerum/vat_tf/blob/master/cnn.py)
    """
    def __init__(self, args):
        super(CNN9, self).__init__()
        input_shape = (3, 32, 32)
        # num_conv = args.num_conv
        num_conv = 128
        affine = args.affine
        self.top_bn = args.top_bn
        self.dropout = args.drop
        self.num_classes = args.num_classes
        # VAT Chainer CNN use bias, TF don't use bias
        self.c1 = nn.Conv2d(input_shape[0], num_conv, 3, 1, 1)
        self.c2 = nn.Conv2d(num_conv, num_conv, 3, 1, 1)
        self.c3 = nn.Conv2d(num_conv, num_conv, 3, 1, 1)
        self.c4 = nn.Conv2d(num_conv, num_conv * 2, 3, 1, 1)
        self.c5 = nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1)
        self.c6 = nn.Conv2d(num_conv * 2, num_conv * 2, 3, 1, 1)
        self.c7 = nn.Conv2d(num_conv * 2, num_conv * 4, 3, 1, 0)
        self.c8 = nn.Conv2d(num_conv * 4, num_conv * 2, 1, 1, 0)
        self.c9 = nn.Conv2d(num_conv * 2, 128, 1, 1, 0)
        # Chainer default eps=2e-05 [Chainer bn](https://docs.chainer.org/en/stable/reference/generated/chainer.links.BatchNormalization.html)
        self.bn1 = nn.BatchNorm2d(num_conv, affine=affine, eps=2e-05)
        self.bn2 = nn.BatchNorm2d(num_conv, affine=affine, eps=2e-05)
        self.bn3 = nn.BatchNorm2d(num_conv, affine=affine, eps=2e-05)
        self.bn4 = nn.BatchNorm2d(num_conv * 2, affine=affine, eps=2e-05)
        self.bn5 = nn.BatchNorm2d(num_conv * 2, affine=affine, eps=2e-05)
        self.bn6 = nn.BatchNorm2d(num_conv * 2, affine=affine, eps=2e-05)
        self.bn7 = nn.BatchNorm2d(num_conv * 4, affine=affine, eps=2e-05)
        self.bn8 = nn.BatchNorm2d(num_conv * 2, affine=affine, eps=2e-05)
        self.bn9 = nn.BatchNorm2d(num_conv, affine=affine, eps=2e-05)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(2, 2)
        # Global average pooling, [batch_size, num_conv, ?, ?] -> [batch_size, num_conv, 1, 1]
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, self.num_classes)
        if self.dropout > 0:
            # make sure it's Dropout, not Dropout2d
            self.dp1 = nn.Dropout(self.dropout)
            self.dp2 = nn.Dropout(self.dropout)

        if self.top_bn:
            self.bnf = nn.BatchNorm1d(self.num_classes, affine=affine, eps=2e-05)
        self.update_bn_stats = True

    def forward(self, x, update_batch_stats=True):
        h = x

        h = self.c1(h)
        h = nfunc.leaky_relu(call_bn(self.bn1, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c2(h)
        h = nfunc.leaky_relu(call_bn(self.bn2, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c3(h)
        h = nfunc.leaky_relu(call_bn(self.bn3, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.mp1(h)
        if self.dropout:
            h = self.dp1(h)

        h = self.c4(h)
        h = nfunc.leaky_relu(call_bn(self.bn4, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c5(h)
        h = nfunc.leaky_relu(call_bn(self.bn5, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c6(h)
        h = nfunc.leaky_relu(call_bn(self.bn6, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.mp2(h)
        if self.dropout:
            h = self.dp2(h)

        h = self.c7(h)
        h = nfunc.leaky_relu(call_bn(self.bn7, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c8(h)
        h = nfunc.leaky_relu(call_bn(self.bn8, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.c9(h)
        h = nfunc.leaky_relu(call_bn(self.bn9, h, update_batch_stats=update_batch_stats), negative_slope=0.1)
        h = self.aap(h)
        output = self.linear(h.view(-1, 128))
        if self.top_bn:
            output = call_bn(self.bnf, output, update_batch_stats=update_batch_stats)
        return output

    def update_batch_stats(self, flag):
        self.update_bn_stats = flag


class CNN3(nn.Module):
    # CNN3 for mnist

    def __init__(self, args):
        super(CNN3, self).__init__()
        self.config = args
        self.top_bn = args.top_bn
        self.dropout_rate = args.drop
        self.num_classes = args.num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=(1, 1), padding=2, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=(1, 1), padding=2, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, self.num_classes, bias=not self.top_bn)
        self.bn1 = nn.BatchNorm2d(32, eps=EPS, momentum=MOMENTUM, affine=args.affine)
        self.bn2 = nn.BatchNorm2d(64, eps=EPS, momentum=MOMENTUM, affine=args.affine)
        if self.top_bn:
            self.bn_fc3 = nn.BatchNorm1d(self.num_classes, eps=EPS, momentum=MOMENTUM, affine=args.affine)

    def forward(self, images, update_batch_stats=True):
        x = images

        # Conv Layer 1
        x = self.conv1(x)
        x = nfunc.relu(x)
        x = nfunc.max_pool2d(x, 2, stride=2)
        x = call_bn(self.bn1, x, update_batch_stats)

        # Conv Layer 2
        x = self.conv2(x)
        x = nfunc.relu(x)
        x = nfunc.max_pool2d(x, 2, stride=2)
        x = call_bn(self.bn2, x, update_batch_stats)

        x = x.view(x.shape[0], -1)

        # fully connect layer 1
        x = nfunc.relu(self.fc1(x))

        # fully connect layer logit
        # x = self.fc2(x)
        if self.top_bn:
            x = call_bn(self.bn_fc3, self.fc2(x), update_batch_stats)
        else:
            x = self.fc2(x)
        return x
