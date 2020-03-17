import os
import argparse
import traceback

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_util
import tensorboardX as tbX

from ExpUtils import *
from torch_func.utils import set_framework_seed, weights_init_normal, adjust_learning_rate, load_checkpoint_by_marker
from torch_func.evaluate import evaluate_classifier
from torch_func.load_dataset import load_dataset
from torch_func.mnist_load_dataset import load_dataset as mnist_load_dataset
import models
from Loss import L0Layer, L0VATOne
from Loss.l0_utils import show_and_save_generated_demo


def parse_args():
    parser = argparse.ArgumentParser(description='L0-VAT Semi-supervised learning in PyTorch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, svhn (default: cifar10)')
    parser.add_argument('--data-dir', type=str, default='data', help='default: data')
    parser.add_argument('--trainer', type=str, default='l0', help='ce, vat, l0(L0, not ten), l02, (default: l0)')
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
    parser.add_argument('--lr-a', type=float, default=0.001, help='learning rate for log_alpha (default: 0.001)')
    parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay used on MNIST (default: 0.95)')
    parser.add_argument('--epoch-decay-start', type=float, default=80, help='start learning rate decay used on SVHN and cifar10 (default: 80)')
    parser.add_argument('--alpha', type=float, default=1, help='alpha for KL div loss (default: 1)')
    parser.add_argument('--eps', type=float, default=1, help='alpha for KL div loss (default: 1)')
    parser.add_argument('--lamb', type=float, default=1.0, help='lambda for unlabeled l0 loss (default: 1)')
    parser.add_argument('--lamb2', type=float, default=0.0, help='lambda for unlabeled smooth loss (default: 1)')
    parser.add_argument('--log-alpha', type=float, default=0.1, help='log alpha initialization of L0VAT (default: 0.1)')
    parser.add_argument('--zeta', type=float, default=1.1, help='zeta for L0VAT, always > 1 (default: 1.1)')
    parser.add_argument('--beta', type=float, default=0.66, help='beta for L0VAT (default: 0.66)')
    parser.add_argument('--gamma', type=float, default=-0.1, help='gamma for L0VAT, always < 0 (default: -0.1)')
    parser.add_argument('--affine', action='store_true', default=False, help='batch norm affine configuration')
    parser.add_argument('--top-bn', action='store_true', default=False, help='enable top batch norm layer')
    parser.add_argument('--ent-min', action='store_true', default=False, help='use entropy minimum')
    parser.add_argument('--kl', type=int, default=1, help='unlabel loss computing, (default: 1)')
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

    if "vat" == args.trainer:
        args.xi = 1e-6
        args.eps = {'mnist': 0.5, 'svhn': 2.5, 'cifar10': 10}[args.dataset]
        args.ent_min = False
        if not args.log_arg:
            args.log_arg = "trainer-eps-xi-ent_min-top_bn-lr"

    if not args.log_arg:
        args.log_arg = "trainer-lr_a-lamb-lamb2-k-top_bn-lr-fig-layer"

    if args.vis:
        args.dir_path = form_dir_path("L0VAT-semi", args)
        set_file_logger(logger, args)
        args.writer = tbX.SummaryWriter(log_dir=args.dir_path)
        os.mkdir("%s/demo" % args.dir_path)
    wlog("args in this experiment:\n%s" % '\n'.join(str(e) for e in sorted(vars(args).items())))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    return args, kwargs


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

    if args.trainer == "l0":
        # shape * 4 / 1024 / 1024 MB, For example, MNIST 60000 * 784 * 4 / 1024 / 1024 ~ 180 MB, CIFAR10 ~ 860 MB
        size, c, h, w = train_ul.data.shape
        if abs(args.log_alpha) > 0.0001:
            if '-np' in args.log_arg:
                train_ul.log_alpha = torch.FloatTensor(np.random.normal(0, args.alpha, (size, 1, h, w))).to(args.device)
            else:
                train_ul.log_alpha = torch.randn((size, 1, h, w), device=args.device)
        else:
            train_ul.log_alpha = torch.zeros((size, 1, h, w)).to(args.device)
        train_ul.log_alpha.requires_grad = True

    args.num_classes = {'mnist': 10, 'svhn': 10, 'cifar10': 10, 'cifar100': 100}[args.dataset]
    test_set = data_util.TensorDataset(torch.FloatTensor(test_set.data), torch.LongTensor(test_set.label))
    test_loader = data_util.DataLoader(test_set, 128, False)
    return train_l, train_ul, test_loader


def init_model(args):
    arch = getattr(models, args.arch)
    model = arch(args)
    if args.debug:
        # weights init is based on numpy, so only need np.random.seed()
        np.random.seed(args.seed)
        model.apply(weights_init_normal)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        exp_marker = "L0VAT-semi/%s" % args.dataset
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
    model.train()
    return model, optimizer, scheduler, start_epoch


def train(dataset_kit, model_kit, args):
    train_l, train_ul, test_loader = dataset_kit
    model, optimizer, scheduler, start_epoch = model_kit

    if "sgd" in args.log_arg:
        opt_method = optim.SGD
    elif "rms" in args.log_arg:
        opt_method = optim.RMSprop
    else:
        opt_method = optim.Adam
    # want to use an optimizer with the history like Adam/RMSprop, it will monitor on all log alpha and take longer time
    # if just monitor on a batch of log alpha, the optimizer will work as SGD
    alpha_opt = opt_method([train_ul.log_alpha], lr=args.lr_a)

    # Define losses.
    criterion = nn.CrossEntropyLoss()

    l0_ins = L0Layer(args)
    l0_ins.train()
    reg_component = L0VATOne(args)

    # show the masked images and masks
    idx = []
    vis_set = None
    if args.vis and "l0" in args.trainer and "fig" in args.log_arg:
        np.random.seed(0)
        for i in range(10):
            ind = np.where(train_ul.label == i)[0]
            idx.append(np.random.choice(ind, 1))
        idx = np.concatenate(idx)
        wlog("select index images %s" % str(list(idx)))
        if args.dataset not in ['mnist', 'svhn']:
            temp, args.data_dir = args.data_dir, 'data1.0'
            _, vis_dataset, _ = get_data(args)
            vis_set = vis_dataset.data[idx]
            args.data_dir = temp
        else:
            vis_set = train_ul.data[idx]
        log_alpha = train_ul.log_alpha[idx]
        show_and_save_generated_demo(l0_ins, log_alpha, train_ul.data[idx], args, 0, 0.1, 1, vis_set=vis_set)

    # train
    saved_count = 0
    start_time = time.time()
    decay_ok = True
    wlog("decay start on generator %s" % str(decay_ok))
    for epoch in range(start_epoch, args.num_epochs):
        for it in range(args.num_batch_it):

            x, t = train_l.get(args.batch_size, aug_trans=args.aug_trans, aug_flip=args.aug_flip)
            images = torch.FloatTensor(x).to(args.device)
            labels = torch.LongTensor(t).to(args.device)

            x_u, t_for_debug, x_u_idx = train_ul.get(args.ul_batch_size, aug_trans=args.aug_trans, aug_flip=args.aug_flip, get_idx=True)
            ul_images = torch.FloatTensor(x_u).to(args.device)

            log_alpha = train_ul.log_alpha[x_u_idx]

            logits = model(images)

            # supervised loss
            ce_loss = criterion(logits, labels)
            sup_loss = ce_loss

            ul_loss = 0
            total_loss = sup_loss
            if "ce" == args.trainer:
                pass
            elif "l0" in args.trainer:
                ul_loss, mask = reg_component(model, ul_images, log_alpha, l0_ins, kl_way=args.kl)
                total_loss += ul_loss
                alpha_opt.zero_grad()
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            total_loss.backward()

            if "l0" == args.trainer and (epoch < args.epoch_decay_start or decay_ok):
                # one stage
                for p in alpha_opt.param_groups[0]['params']:
                    if p.grad is None:
                        continue
                    p.grad.data = -p.grad.data
                alpha_opt.step()
            optimizer.step()

            if ((epoch % args.log_interval) == 0 and it == args.num_batch_it - 1) or (args.debug and it < 5 and epoch == 0):
                n_err, test_loss = evaluate_classifier(model, test_loader, args.device)
                acc = 1 - n_err / len(test_loader.dataset)
                cost_time = time.time() - start_time
                wlog("Epoch: %d Train Loss %.4f ce %.5f, ul loss %.5f, test loss %.5f, test acc %.4f, time %.2f" % (epoch, total_loss, ce_loss, ul_loss, test_loss, acc, cost_time))

                if args.vis and it == args.num_batch_it - 1:
                    if (epoch + 1) * 10 % args.num_epochs == 0:
                        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': test_loss, 'acc': acc},
                                   "%s/model.pth" % args.dir_path)
                        if epoch > args.num_epochs / 2:
                            shutil.copy("%s/model.pth" % args.dir_path, "%s/model_%d.pth" % (args.dir_path, epoch+1))
                    pred_y = torch.max(logits, dim=1)[1]
                    train_acc = 1.0 * torch.sum(pred_y == labels).item() / pred_y.shape[0]

                    dicts = {"Train/CELoss": ce_loss, "Train/UnsupLoss": ul_loss, "Train/Loss": total_loss, "Test/Acc": acc, "Test/Loss": test_loss, "Train/Acc": train_acc}
                    vis_step(args.writer, epoch, dicts)

                    save_interval = 5
                    if args.epoch_decay_start + 10 > epoch >= args.epoch_decay_start:
                        save_interval = 1
                    if "l0" in args.trainer and epoch % save_interval == 0 and "fig" in args.log_arg:
                        log_alpha = train_ul.log_alpha[idx]
                        show_and_save_generated_demo(l0_ins, log_alpha, train_ul.data[idx], args, epoch+1, acc, saved_count, vis_set)
                        saved_count += 1
                start_time = time.time()

        if scheduler == "linear":
            lr = adjust_learning_rate(optimizer, epoch, args)
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]
        if epoch % args.log_interval == 0:
            wlog("learning rate %f" % lr)
            if args.vis:
                args.writer.add_scalar("Optimizer/LearningRate", lr, epoch)


def main(args):
    """Training function."""
    set_framework_seed(args.seed, args.debug)
    dataset_kit = get_data(args)
    model_kit = init_model(args)
    train(dataset_kit, model_kit, args)


if __name__ == "__main__":
    arg, kwarg = parse_args()
    # noinspection PyBroadException
    try:
        main(arg)
    except KeyboardInterrupt:
        if arg.dir_path:
            os.rename(arg.dir_path, arg.dir_path + "_stop")
    except BaseException as err:

        traceback.print_exc()
        if arg.dir_path:
            shutil.rmtree(arg.dir_path)
