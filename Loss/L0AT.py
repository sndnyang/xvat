import torch.nn as nn

from torch_func.utils import l2_normalize


class L0AT(object):

    def __init__(self, args):
        self.args = args
        self.eps = args.eps
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, model, image, label):
        image.requires_grad = True
        logits = model(image)

        loss = self.cross_entropy(logits, label)
        loss.backward()
        if "sign" in self.args.log_arg:
            d = image.grad.data.clone().sign()
        else:
            d = l2_normalize(image.grad)

        logits_x_hat = model(image.detach() + self.eps * d)
        loss_at = self.cross_entropy(logits_x_hat, label)

        return loss_at, d
