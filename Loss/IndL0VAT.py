import torch
import torch.nn as nn
import torch.nn.functional as nfunc

from torch_func.utils import entropy


def compute_lds(model, perturbations, criterion, prob_x, log_prob_x, kl_way=1):
    try:
        logits_x_hat = model(perturbations, update_batch_stats=False)
    except TypeError:
        logits_x_hat = model(perturbations)
    if kl_way == 1:
        prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
        train_lds = torch.mean(criterion(log_prob_x, prob_x_hat).sum(dim=1))
    else:
        log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
        train_lds = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))
    return train_lds


class IndL0VAT(object):

    def __init__(self, args):
        self.args = args
        self.device = args.device
        # PyTorch Version difference
        try:
            self.dis_criterion = nn.KLDivLoss(reduction='none')
        except TypeError:
            self.dis_criterion = nn.KLDivLoss(size_average=False, reduce=False)

    def __call__(self, model, images, generator, l0_ins, kl_way=1):
        l0_ins.train()

        # find mask to maximize the kl divergence and l0 loss of mask
        logits = model(images, update_batch_stats=False)
        prob_x = nfunc.softmax(logits.detach(), dim=1)
        log_prob_x = nfunc.log_softmax(logits.detach(), dim=1)

        log_alpha = generator(images)
        for i in range(self.args.k):
            perturbations, train_mask = l0_ins.mask_filter(images, log_alpha)
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

        if "reu" in self.args.log_arg or self.args.k == 0:
            pert_mask = l0_ins.get_mask(log_alpha, re_sample=True)
        else:
            pert_mask = l0_ins.get_mask(log_alpha, re_sample=False)
        perturbations = images * pert_mask.detach()
        lds = compute_lds(model, perturbations, self.dis_criterion, prob_x, log_prob_x, kl_way=kl_way)
        if 'ent' in self.args.trainer:
            lds = lds + entropy(logits)
        return lds, pert_mask
