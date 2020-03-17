import torch
import torch.nn as nn
import torch.nn.functional as nfunc
import numpy as np

from ExpUtils import wlog
from .L0Layer import init_smooth_conv


class AutoL0Layer(nn.Module):
    def __init__(self, args):
        super(AutoL0Layer, self).__init__()
        # gamma = -0.1
        # zeta = 1.1
        # beta = 0.66
        self.args = args
        self.device = args.device
        self.constant_eps = 1e-10
        self.gamma = args.gamma
        self.zeta = args.zeta
        self.beta = args.beta
        self.eps = None
        self.const1 = self.beta * np.log(-self.gamma / self.zeta + self.constant_eps)
        self.loss3conv1, self.loss3conv2, self.loss3conv3, self.loss3conv4 = init_smooth_conv(args.device)
        self.u = None
        wlog("eps = 0 with clip and eps > 1 no clip no upper/lower bound")

    def get_mask(self, log_alpha, re_sample=True):
        if self.training:
            if self.u is None or re_sample:
                self.u = u = torch.rand(log_alpha.size(), device=self.args.device)
            else:
                u = self.u
            s = torch.sigmoid((torch.log(u + self.constant_eps) - torch.log(1 - u + self.constant_eps) + log_alpha + self.constant_eps) / self.beta)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            mask = nfunc.hardtanh(s_bar, min_val=0, max_val=1)
        else:
            s = torch.sigmoid(log_alpha / self.beta)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            mask = nfunc.hardtanh(s_bar, min_val=0, max_val=1)
        return mask

    def mask_filter(self, images, log_alpha):
        masks = self.get_mask(log_alpha)
        if self.eps < 0.001:
            p_max = torch.abs(images).max().item()
            shape = [images.shape[0]] + [1] * (len(images.shape) - 1)
            perturbations = images * masks
            perturbations = p_max * perturbations / torch.max(torch.abs(perturbations.view(images.shape[0], -1)), dim=1)[0].view(shape)
        elif self.eps < 0.001:
            perturbations = images * masks
        else:
            perturbations = self.eps * images * masks
            # if p_max < 1.001:
            # cifar with ZCA whitening will have values whose range is about [-30, 30]
            p_min, p_max = images.min().item(), images.max().item()
            perturbations = torch.clamp(perturbations, p_min, p_max)
        return perturbations, masks

    def l0_loss(self, log_alpha):
        loss = torch.mean(torch.sigmoid(self.eps * log_alpha - self.const1))
        return loss

    def smooth_loss(self, mask):
        mask = torch.sigmoid(mask - self.const1)
        diff1 = torch.mean(torch.abs(self.loss3conv1(mask)))
        diff2 = torch.mean(torch.abs(self.loss3conv2(mask)))
        diff3 = torch.mean(torch.abs(self.loss3conv3(mask)))
        diff4 = torch.mean(torch.abs(self.loss3conv4(mask)))
        return diff1 + diff2 + diff3 + diff4
