import torch
import torch.nn as nn
import torch.nn.functional as nfunc
import numpy as np

from ExpUtils import wlog


def init_smooth_conv(device):
    """
    :param device: GPU/CPU device
    :return: Smooth Conv Component:
        [[ 1],   [[1, -1]], [[1,  0],  [[ 0, 1],
         [-1]],              [0, -1]],  [-1, 0]]
    """
    weights1 = torch.randn(1, 1, 2, 1, requires_grad=False)
    weights1[:, :, 0, 0] = 1
    weights1[:, :, 1, 0] = -1
    loss3conv1 = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=0, bias=False)
    loss3conv1.weight = nn.Parameter(weights1)
    for p in loss3conv1.parameters():
        p.requires_grad = False

    weights2 = torch.randn(1, 1, 1, 2, requires_grad=False)
    weights2[:, :, 0, 0] = 1
    weights2[:, :, 0, 1] = -1
    loss3conv2 = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=0, bias=False)
    loss3conv2.weight = nn.Parameter(weights2)
    for p in loss3conv2.parameters():
        p.requires_grad = False

    weights3 = torch.Tensor([[1, 0], [0, -1]]).unsqueeze(0).unsqueeze(0)
    loss3conv3 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False)
    loss3conv3.weight = nn.Parameter(weights3)
    for p in loss3conv3.parameters():
        p.requires_grad = False

    weights4 = torch.Tensor([[0, 1], [-1, 0]]).unsqueeze(0).unsqueeze(0)
    loss3conv4 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False)
    loss3conv4.weight = nn.Parameter(weights4)
    for p in loss3conv4.parameters():
        p.requires_grad = False
    loss3conv1 = loss3conv1.to(device)
    loss3conv2 = loss3conv2.to(device)
    loss3conv3 = loss3conv3.to(device)
    loss3conv4 = loss3conv4.to(device)
    return loss3conv1, loss3conv2, loss3conv3, loss3conv4


class L0Layer(nn.Module):
    def __init__(self, args):
        super(L0Layer, self).__init__()
        # gamma = -0.1
        # zeta = 1.1
        # beta = 0.66
        self.args = args
        self.device = args.device
        self.constant_eps = 1e-10
        self.gamma = args.gamma
        self.zeta = args.zeta
        self.beta = args.beta
        self.const1 = self.beta * np.log(-self.gamma / self.zeta + self.constant_eps)
        self.loss3conv1, self.loss3conv2, self.loss3conv3, self.loss3conv4 = init_smooth_conv(args.device)
        self.u = None
        self.clip = False
        self.eps = args.eps
        wlog("eps = 0 eps > 1 with clip bound %s" % str(self.clip))

    def get_mask(self, log_alpha, re_sample=True):
        if self.training:
            if self.u is None or re_sample:
                self.u = u = torch.rand(log_alpha.size(), device=self.args.device)
            else:
                u = self.u
            s_u = torch.log(u + self.constant_eps) - torch.log(1 - u + self.constant_eps)
            s = torch.sigmoid((s_u + log_alpha + self.constant_eps) / self.beta)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            mask = nfunc.hardtanh(s_bar, min_val=0, max_val=1)
        else:
            s = torch.sigmoid(log_alpha / self.beta)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            mask = nfunc.hardtanh(s_bar, min_val=0, max_val=1)
        if self.args.dataset in ["image", 'imagenet']:
            mask = nfunc.interpolate(mask, size=(224, 224), mode='nearest')
        if self.args.dataset in ["image64", 'imagenet64']:
            if 'pool' in self.args.layer:
                mask = nfunc.interpolate(mask, size=(64, 64), mode='nearest')
        if 'caltech' in self.args.dataset:
            if 'pool' in self.args.layer:
                if "incept" == self.args.arch:
                    mask = nfunc.interpolate(mask, size=(299, 299), mode='nearest')
                else:
                    mask = nfunc.interpolate(mask, size=(256, 256), mode='nearest')
        return mask

    def mask_filter(self, images, log_alpha):
        masks = self.get_mask(log_alpha)
        if self.eps < 0.0001:
            p_max = torch.abs(images).max().item()
            shape = [images.shape[0]] + [1] * (len(images.shape) - 1)
            perturbations = images * masks
            perturbations = p_max * perturbations / torch.max(torch.abs(perturbations.view(images.shape[0], -1)), dim=1)[0].view(shape)
        else:
            perturbations = self.eps * images * masks
        if self.clip:
            p_min, p_max = images.min().item(), images.max().item()
            perturbations = torch.clamp(perturbations, p_min - 0.1, p_max + 0.1)
        return perturbations, masks

    def l0_loss(self, log_alpha):
        loss = torch.mean(torch.sigmoid(log_alpha - self.const1))
        return loss

    def smooth_loss(self, mask):
        mask = torch.sigmoid(mask - self.const1)
        diff1 = torch.mean(torch.abs(self.loss3conv1(mask)))
        diff2 = torch.mean(torch.abs(self.loss3conv2(mask)))
        diff3 = torch.mean(torch.abs(self.loss3conv3(mask)))
        diff4 = torch.mean(torch.abs(self.loss3conv4(mask)))
        return diff1 + diff2 + diff3 + diff4
