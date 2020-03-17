import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import models.GeneratorNN

from torch_func.utils import weights_init_normal, weights_init_uniform


class TransductiveGenerator:
    """
    Randomly generate log_alpha(s) for each image in a transductive way.
    Just store in train_set.log_alpha and it works well. So this class is not used...
    """
    def __init__(self, train_set, args):
        if "adam" in args.log_arg:
            opt_method = optim.Adam
        elif "rms" in args.log_arg:
            opt_method = optim.RMSprop
        else:
            opt_method = optim.SGD

        # shape * 4 / 1024 / 1024 MB, For example, MNIST 60000 * 784 * 4 / 1024 / 1024 ~ 180 MB, CIFAR10 ~ 860 MB
        self.args = args
        size, c, h, w = train_set.data.shape
        if abs(args.alpha) > 0.0001:
            self.log_alpha = torch.FloatTensor(np.random.normal(0, args.alpha, (size, 1, h, w))).to(args.device)
        else:
            self.log_alpha = torch.zeros((size, 1, h, w)).to(args.device)
        self.optimizer = None
        self.index = None
        self.opt_method = opt_method

    def __call__(self, images, index=None):
        assert index is not None
        log_alpha = self.log_alpha[index].to(self.args.device)
        log_alpha.requires_grad = True
        self.optimizer = self.opt_method([log_alpha], lr=self.args.lr_a)
        self.index = index
        return log_alpha

    def step(self, log_alpha):
        self.optimizer.step()
        self.log_alpha[self.index] = log_alpha.detach()


class InductiveGenerator(nn.Module):
    """
    Generate log_alpha from images by an inductive way
    """
    def __init__(self, args):
        super(InductiveGenerator, self).__init__()

        if args.dataset == "mnist":
            c = 1
            conv_arch = "CNN" + str(args.layer) + "l"
        elif args.dataset == "svhn":
            c = 3
            conv_arch = "CNN" + str(args.layer) + "l"
        elif args.dataset == "cifar10":
            c = 3
            conv_arch = "CNN" + str(args.layer) + "l"
        elif args.dataset == "cifar100":
            c = 3
            conv_arch = "CNN" + str(args.layer) + "l"
        elif args.dataset in ['image', 'imagenet']:
            c = 3
            conv_arch = "CNN" + str(args.layer)
        elif args.dataset in ['image64', 'imagenet64']:
            c = 3
            conv_arch = "CNN" + str(args.layer)
        elif args.dataset in ['1', '2']:
            c = 1
            conv_arch = 'MLP1'
        else:
            c = 3
            conv_arch = "CNN" + str(args.layer)

        self.args = args
        cnn = getattr(models.GeneratorNN, conv_arch)
        self.conv = cnn(channel=c)
        if "normal" in args.log_arg:
            self.conv.apply(weights_init_normal)
        elif "uni" in args.log_arg:
            self.conv.apply(weights_init_uniform)
        self.conv.to(args.device)

    def __call__(self, images):
        log_alpha = self.conv(images)
        return log_alpha
