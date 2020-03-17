import torch.nn as nn
import torch.nn.functional as nfunc

from torch_func.utils import entropy
from .IndL0VAT import compute_lds


class IndL0ATOne(object):

    def __init__(self, args):
        self.args = args
        self.device = args.device
        # PyTorch Version difference
        try:
            self.dis_criterion = nn.KLDivLoss(reduction='none')
        except TypeError:
            self.dis_criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, model, images, label, generator, l0_ins, dis="ce"):
        l0_ins.train()

        # find mask to maximize the kl divergence and l0 loss of mask
        try:
            logits = model(images, update_batch_stats=False)
        except TypeError:
            logits = model(images)
        prob_x = nfunc.softmax(logits.detach(), dim=1)
        log_prob_x = nfunc.log_softmax(logits.detach(), dim=1)

        log_alpha = generator(images)
        perturbations, train_mask = l0_ins.mask_filter(images, log_alpha)

        # l0 loss of image
        l0_loss = 0
        if abs(self.args.lamb - 0) > 0.001:
            l0_loss = l0_ins.l0_loss(log_alpha)

        lds = self.args.lamb * l0_loss

        if dis == "kl":
            lds += self.args.alpha * compute_lds(model, perturbations, self.dis_criterion, prob_x, log_prob_x, kl_way=1)
        else:
            logits_x_hat = model(perturbations)
            lds = lds + self.args.alpha * self.cross_entropy(logits_x_hat, label)

        return lds, perturbations
