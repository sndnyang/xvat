import time

import torch
import torch.nn as nn
import torch.nn.functional as nfunc

from ExpUtils import wlog


def evaluate_classifier(classifier, loader, device):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert isinstance(device, torch.device)

    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    n_err = 0
    t_loss = 0
    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            logits = classifier(x.to(device))
            loss = criterion(logits, y).item()
            t_loss += loss * y.shape[0]
            prob_y = nfunc.softmax(logits, dim=1)
            pred_y = torch.max(prob_y, dim=1)[1]
            n_err += torch.sum(pred_y != y).item()

    classifier.train()

    return n_err, t_loss / len(loader.dataset)


def mse(p, q):
    criterion = nn.MSELoss()
    with torch.no_grad():
        loss = criterion(nfunc.softmax(p, dim=1), nfunc.softmax(q, dim=1)).item()
    return loss


def kl_div(p, q):
    criterion = nn.KLDivLoss(reduction="batchmean")
    with torch.no_grad():
        loss = criterion(nfunc.log_softmax(p, dim=1), nfunc.softmax(q, dim=1)).item()
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.val = 0
        self.name = name
        self.fmt = fmt
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmt_str = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmt_str.format(**self.__dict__)


def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate_imagenet(model, val_loader, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    criterion = nn.CrossEntropyLoss()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, top_k=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        wlog(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}' .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
