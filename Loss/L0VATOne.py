import torch.nn as nn
import torch.nn.functional as nfunc

from torch_func.utils import entropy
from .IndL0VAT import compute_lds


class L0VATOne(object):

    def __init__(self, args):
        self.args = args
        self.device = args.device
        # PyTorch Version difference
        try:
            self.dis_criterion = nn.KLDivLoss(reduction='none')
        except TypeError:
            self.dis_criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.use_ent_min = args.ent_min

    def __call__(self, model, images, log_alpha, l0_ins, kl_way=1):
        l0_ins.train()

        # find mask to maximize the kl divergence and l0 loss of mask
        logits = model(images, update_batch_stats=False)
        prob_x = nfunc.softmax(logits.detach(), dim=1)
        log_prob_x = nfunc.log_softmax(logits.detach(), dim=1)

        perturbations, train_mask = l0_ins.mask_filter(images, log_alpha)

        # l0 loss of image
        l0_loss = 0
        if abs(self.args.lamb - 0) > 0.001:
            l0_loss = l0_ins.l0_loss(log_alpha)

        smooth_loss = 0
        if abs(self.args.lamb2 - 0) > 0.001:
            smooth_loss = l0_ins.smooth_loss(train_mask)

        lds = self.args.lamb * l0_loss + self.args.lamb2 * smooth_loss
        lds += self.args.alpha * compute_lds(model, perturbations, self.dis_criterion, prob_x, log_prob_x, kl_way=kl_way)

        if 'ent' in self.args.trainer:
            lds = lds + entropy(logits)

        return lds, train_mask
