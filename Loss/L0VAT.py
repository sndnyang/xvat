import torch
import torch.nn as nn
import torch.nn.functional as nfunc

from torch_func.utils import entropy


class L0VAT(object):

    def __init__(self, args):
        self.args = args
        self.device = args.device
        # PyTorch Version difference
        try:
            self.dis_criterion = nn.KLDivLoss(reduction='none')
        except TypeError:
            self.dis_criterion = nn.KLDivLoss(size_average=False, reduce=False)

    def __call__(self, model, images, log_alphas, index, l0_ins, optimizer, kl_way=1):
        l0_ins.train()
        log_alpha = log_alphas[index].to(images.device)
        # log_alpha.requires_grad = True

        # find mask to maximize the kl divergence and l0 loss of mask
        logits = model(images, update_batch_stats=False)
        prob_x = nfunc.softmax(logits.detach(), dim=1)
        log_prob_x = nfunc.log_softmax(logits.detach(), dim=1)

        for i in range(self.args.k):
            perturbations, train_mask = l0_ins.mask_filter(images, log_alpha)
            logits_x_hat = model(perturbations, update_batch_stats=False)
            if kl_way == 1:
                prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
                train_lds = torch.mean(self.dis_criterion(log_prob_x, prob_x_hat).sum(dim=1))
            else:
                log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
                train_lds = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))

            # l0 loss of image
            l0_loss = l0_ins.l0_loss(log_alpha)

            train_lds += self.args.lamb * l0_loss
            if abs(self.args.lamb2 - 0) > 0.001:
                smooth_loss = l0_ins.smooth_loss(train_mask)
                train_lds -= self.args.lamb2 * smooth_loss

            loss = -train_lds
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # log_alphas[index] = log_alpha.detach()

        if "reu" in self.args.log_arg or self.args.k == 0:
            pert_mask = l0_ins.get_mask(log_alpha, re_sample=True)
        else:
            pert_mask = l0_ins.get_mask(log_alpha, re_sample=False)
        logits_x_hat = model(images * pert_mask, update_batch_stats=False)
        if kl_way == 1:
            prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
            lds = torch.mean(self.dis_criterion(log_prob_x, prob_x_hat).sum(dim=1))
        else:
            log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
            lds = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))
        if 'ent' in self.args.trainer:
            lds = lds + entropy(logits)
        return lds, pert_mask
