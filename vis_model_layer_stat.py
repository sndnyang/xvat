import os
import argparse
import traceback

import torch.optim as optim
import torch.utils.data as data_util
import tensorboardX as tbX

from ExpUtils import *
from torch_func.utils import set_framework_seed, weights_init_normal, load_checkpoint_by_marker
from torch_func.evaluate import evaluate_classifier
from torch_func.load_dataset import load_dataset
from torch_func.mnist_load_dataset import load_dataset as mnist_load_dataset
import models

from vis import *


def parse_args():
    parser = argparse.ArgumentParser(description='VAT Semi-supervised learning in PyTorch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, svhn (default: cifar10)')
    parser.add_argument('--data-dir', type=str, default='data', help='default: data')
    parser.add_argument('--trainer', type=str, default='vat', help='ce, vat (default: vat)')
    parser.add_argument('--size', type=int, default=4000, help='size of training data set, fixed size for datasets (default: 4000)')
    parser.add_argument('--arch', type=str, default='CNN9', help='CNN9 for semi supervised learning on dataset')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='N', help='number of epochs (default: 100)')
    parser.add_argument('--num-batch-it', type=int, default=400, metavar='N', help='number of batch iterations (default: 400)')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default="", metavar='N', help='gpu id list (default: auto select)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='iterations to wait before logging status, (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size of training data set, MNIST uses 100 (default: 32)')
    parser.add_argument('--ul-batch-size', type=int, default=128, help='batch size of unlabeled data set, MNIST uses 250 (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay used on MNIST (default: 0.95)')
    parser.add_argument('--epoch-decay-start', type=float, default=80, help='start learning rate decay used on SVHN and cifar10 (default: 80)')
    parser.add_argument('--xi', type=float, default=1e-6, help='xi for VAT loss (default: 1e-6)')
    parser.add_argument('--eps', type=float, default=1.0, help='epsilon for VAT loss (default: 1.0)')
    parser.add_argument('--ent-min', action='store_true', default=False, help='use entropy minimum')
    parser.add_argument('--affine', action='store_true', default=False, help='batch norm affine configuration')
    parser.add_argument('--top-bn', action='store_true', default=False, help='enable top batch norm layer')
    parser.add_argument('--k', type=int, default=1, help='optimization times, (default: 1)')
    parser.add_argument('--kl', type=int, default=0, help='unlabel loss computing, (default: 0)')
    parser.add_argument('--aug-trans', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--aug-flip', action='store_true', default=False, help='data augmentation flip')
    parser.add_argument('--drop', type=float, default=0.5, help='dropout rate, (default: 0.5)')
    parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory, (default: an absolute path)')
    parser.add_argument('--log-arg', type=str, default='', metavar='S', help='show the arguments in directory name')
    parser.add_argument('--debug', action='store_true', default=False, help='compare log side by side')
    parser.add_argument('--vis', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('-r', '--resume', type=str, default='', metavar='S', help='resume from pth file')

    args = parser.parse_args()
    args.dir_path = None

    if args.gpu_id == "":
        args.gpu_id = auto_select_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if not args.log_arg:
        args.log_arg = "trainer-eps-xi-ent_min-top_bn-lr"

    if args.vis:
        args.dir_path = form_dir_path("L0VAT-semi", args)
        set_file_logger(logger, args)
        args.writer = tbX.SummaryWriter(log_dir=args.dir_path)
        os.mkdir("%s/demo" % args.dir_path)
    wlog("args in this experiment:\n%s" % '\n'.join(str(e) for e in sorted(vars(args).items())))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def get_data(args):
    if args.dataset == "mnist":
        # VAT parameters for MNIST. They are different from SVHN/CIFAR10
        if args.size != 1000:
            args.size = 100
        args.num_batch_it = 500
        train_l, train_ul, test_set = mnist_load_dataset("mnist", size=args.size, keep=True)
    else:
        train_l, train_ul, test_set = load_dataset("%s/%s" % (args.data_dir, args.dataset), valid=False, dataset_seed=args.seed)
    wlog("N_train_labeled:{}, N_train_unlabeled:{}".format(train_l.size, train_ul.size))
    wlog("train_l sum {}".format(train_l.data.sum()))
    summary(train_ul.data)

    args.num_classes = {'mnist': 10, 'svhn': 10, 'cifar10': 10, 'cifar100': 100}[args.dataset]
    test_set = data_util.TensorDataset(torch.FloatTensor(test_set.data), torch.LongTensor(test_set.label))
    test_loader = data_util.DataLoader(test_set, 128, False)
    return train_l, train_ul, test_loader


def init_model(args, resume_dir='L0VAT-semi'):
    arch = getattr(models, args.arch)
    model = arch(args)
    if args.debug:
        # weights init is based on numpy, so only need np.random.seed()
        np.random.seed(args.seed)
        model.apply(weights_init_normal)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        exp_marker = "%s/%s" % (resume_dir, args.dataset)
        checkpoint = load_checkpoint_by_marker(args, exp_marker)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
    if args.dataset == "mnist":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    else:
        scheduler = "linear"
    model = model.to(args.device)
    model.eval()
    return model, optimizer, scheduler, start_epoch


def main(args):
    """Training function."""
    set_framework_seed(args.seed, args.debug)
    dataset_kit = get_data(args)
    model_kit = init_model(args)
    model = model_kit[0]
    run_time = args.dataset + ' ' + args.resume + " " + time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    err, _ = evaluate_classifier(model, dataset_kit[2], args.device)
    acc = 1 - err / len(dataset_kit[2].dataset)
    print(acc)
    filters = model.c1
    if isinstance(filters, torch.nn.Conv2d):
        w = filters.weight.cpu().detach().numpy().transpose((0, 2, 3, 1))
        b = filters.bias.cpu().detach().numpy()
    else:
        raise NotImplementedError
    vis_conv_filters(w, b, model_kit[3], run_time, acc, "1")

    filters = model.c2
    if isinstance(filters, torch.nn.Conv2d):
        w = filters.weight.cpu().detach().numpy().reshape(-1, 8, 16)
        b = filters.bias.cpu().detach().numpy()
    else:
        raise NotImplementedError
    vis_conv_filters(w, b, model_kit[3], run_time, acc, "2")

    filters = model.c5
    if isinstance(filters, torch.nn.Conv2d):
        w = filters.weight.cpu().detach().numpy().reshape(-1, 16, 16)
        b = filters.bias.cpu().detach().numpy()
    else:
        raise NotImplementedError
    vis_conv_filters(w, b, model_kit[3], run_time, acc, "5")

    filters = model.c9
    if isinstance(filters, torch.nn.Conv2d):
        w = filters.weight.cpu().detach().numpy().reshape(-1, 16, 16)
        b = filters.bias.cpu().detach().numpy()
    else:
        raise NotImplementedError
    vis_conv_filters(w, b, model_kit[3], run_time, acc, "9")


if __name__ == "__main__":
    arg = parse_args()
    # noinspection PyBroadException
    try:
        main(arg)
    except BaseException as err:

        traceback.print_exc()
        if arg.dir_path:
            shutil.rmtree(arg.dir_path)
