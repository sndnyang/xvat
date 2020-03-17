import torch
import torch.nn as nn
import torch.nn.functional as nfunc
import numpy as np

from torch_func.utils import l2_normalize, entropy


class VAT(object):

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.xi = args.xi
        self.eps = args.eps
        self.k = args.k
        self.debug = args.debug
        # PyTorch Version difference
        try:
            self.kl_div = nn.KLDivLoss(reduction='none')
        except TypeError:
            self.kl_div = nn.KLDivLoss(size_average=False, reduce=False)

    def __call__(self, model, image, kl_way=0, return_adv=False):
        try:
            logits = model(image, update_batch_stats=False)
        except TypeError:
            logits = model(image)

        prob_x = nfunc.softmax(logits.detach(), dim=1)
        log_prob_x = nfunc.log_softmax(logits.detach(), dim=1)
        if self.debug:
            # np generator is more controllable than torch.randn(image.size())
            d = np.random.standard_normal(image.size())
            d = l2_normalize(torch.FloatTensor(d).to(self.device))
        else:
            d = torch.randn(image.size(), device=self.args.device)
            d = l2_normalize(d)

        for ip in range(self.k):
            d *= self.xi
            d.requires_grad = True
            t = image.detach()
            x_hat = t + d
            try:
                logits_x_hat = model(x_hat, update_batch_stats=False)
            except TypeError:
                logits_x_hat = model(x_hat)
            if kl_way == 1:
                prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
                adv_distance = torch.mean(self.kl_div(log_prob_x, prob_x_hat).sum(dim=1))
            else:
                # official theano code compute in this way
                log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
                adv_distance = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))
            # use backward, compute grad on Image and Model weight, use grad, not compute grad on model's weight.
            adv_distance.backward()
            grad_x_hat = d.grad     # / self.xi
            # grad_x_hat = torch.autograd.grad(adv_distance, d)[0]
            # scale it or not, since it will pass through _l2_normalize, but maybe numerical issues
            d = l2_normalize(grad_x_hat).to(self.device)

        try:
            logits_x_hat = model(image + self.eps * d, update_batch_stats=False)
        except TypeError:
            logits_x_hat = model(image + self.eps * d)
        if kl_way == 1:
            prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
            lds = torch.mean(self.kl_div(log_prob_x, prob_x_hat).sum(dim=1))
        else:
            # official theano code works in this way
            log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
            lds = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))

        if 'ent' in self.args.trainer:
            lds = lds + entropy(logits)

        if return_adv:
            return lds, self.eps * d
        else:
            return lds
