import torch
import torch.nn as nn
import torch.nn.functional as nfunc
import numpy as np

from torch_func.utils import l2_normalize, entropy


from .IndL0VAT import compute_lds


class AXVAT(object):

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.eps = args.eps
        self.debug = args.debug
        self.k = 1
        self.xi = 1e-6
        # PyTorch Version difference
        try:
            self.dis_criterion = nn.KLDivLoss(reduction='none')
        except TypeError:
            self.dis_criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.use_ent_min = args.ent_min

    def __call__(self, model, images, generator, l0_ins, kl_way=1):
        l0_ins.train()

        # find mask to maximize the kl divergence and l0 loss of mask
        logits = model(images, update_batch_stats=False)
        prob_x = nfunc.softmax(logits.detach(), dim=1)
        log_prob_x = nfunc.log_softmax(logits.detach(), dim=1)
        if self.debug:
            # np generator is more controllable than torch.randn(image.size())
            d = np.random.standard_normal(images.size())
            d = l2_normalize(torch.FloatTensor(d).to(self.device))
        else:
            d = torch.randn(images.size(), device=self.args.device)
            d = l2_normalize(d)

        for ip in range(self.k):
            d *= self.xi
            d.requires_grad = True
            t = images.detach()
            x_hat = t + d
            logits_x_hat = model(x_hat, update_batch_stats=False)
            if kl_way == 1:
                prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
                adv_distance = torch.mean(self.dis_criterion(log_prob_x, prob_x_hat).sum(dim=1))
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

        add_pert = images + self.eps * d
        logits_x_hat = model(add_pert, update_batch_stats=False)
        if kl_way == 1:
            prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
            add_lds = torch.mean(self.dis_criterion(log_prob_x, prob_x_hat).sum(dim=1))
        else:
            # official theano code works in this way
            log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
            add_lds = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))

        log_alpha = generator(add_pert)
        for i in range(self.args.k):
            train_mask = l0_ins.get_mask(log_alpha)
            perturbations = 2 * add_pert * train_mask
            train_lds = compute_lds(model, perturbations, self.dis_criterion, prob_x, log_prob_x, kl_way=kl_way)

            # l0 loss of image
            l0_loss = 0
            if abs(self.args.lamb - 0) > 0.001:
                l0_loss = l0_ins.l0_loss(log_alpha)

            smooth_loss = 0
            if abs(self.args.lamb2 - 0) > 0.001:
                smooth_loss = l0_ins.smooth_loss(train_mask)

            loss = -(train_lds + self.args.lamb * l0_loss + self.args.lamb2 * smooth_loss)
            generator.optimizer.zero_grad()
            loss.backward()
            generator.optimizer.step()
            log_alpha = generator(images)

        pert_mask = l0_ins.get_mask(log_alpha, re_sample=True)
        perturbations = images * pert_mask.detach()
        mul_lds = compute_lds(model, perturbations, self.dis_criterion, prob_x, log_prob_x, kl_way=kl_way)

        # alpha, 1-alpha ?  Or  add_lds is the baseline(*1), then add alpha * mul_lds
        lds = add_lds + self.args.alpha * mul_lds

        if 'ent' in self.args.trainer:
            lds = lds + entropy(logits)

        return lds, pert_mask
