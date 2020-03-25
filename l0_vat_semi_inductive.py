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
from Loss import L0Layer, IndL0VATOne
from Loss.l0_utils import *
from Loss.LocationGenerator import InductiveGenerator
from Loss.VAT import VAT


def parse_args():
    parser = argparse.ArgumentParser(description='L0-VAT Inductive Semi-supervised learning in PyTorch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, svhn (default: cifar10)')
    parser.add_argument('--data-dir', type=str, default='data', help='default: data')
    parser.add_argument('--trainer', type=str, default='inl0', help='ce, vat, inl0(L0, not ten), (default: inl0)')
    parser.add_argument('--size', type=int, default=4000, help='size of training data set, fixed size for datasets (default: 4000)')
    parser.add_argument('--arch', type=str, default='CNN9', help='CNN9 for semi supervised learning on dataset')
    parser.add_argument('--layer', type=str, default='2', help='the layer of CNN used by the generator, (default: 2)')
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
    parser.add_argument('--lamb2', type=float, default=0.0, help='lambda for unlabeled smooth loss (default: 0)')
    parser.add_argument('--l2-lamb', type=float, default=5e-4, help='lambda for L2 norm (default: )')
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

    if not args.log_arg:
        args.log_arg = "trainer-lr_a-lamb-lamb2-k-top_bn-lr-fig-layer"

    if args.vis:
        args.dir_path = form_dir_path("L0VAT-semi", args)
        set_file_logger(logger, args)
        args.writer = tbX.SummaryWriter(log_dir=args.dir_path)
        os.mkdir("%s/demo" % args.dir_path)
    wlog("args in this experiment:\n%s" % '\n'.join(str(e) for e in sorted(vars(args).items())))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
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

    # for MNIST, use 2-layer CNN is much better than 1-layer CNN as the generator
    generator = InductiveGenerator(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        exp_marker = "L0VAT-semi/%s" % args.dataset
        checkpoint = load_checkpoint_by_marker(args, exp_marker)
        model.load_state_dict(checkpoint['model_state_dict'])
        generator.conv.load_state_dict(checkpoint['generator_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)

    if 'inl0' not in args.trainer and 'ax' not in args.trainer:
        opt_kit = optimizer
    else:
        if "sgd" in args.log_arg:
            alpha_opt = optim.SGD(generator.parameters(), lr=args.lr_a, momentum=0.1)
        elif "rms" in args.log_arg:
            alpha_opt = optim.RMSprop(generator.parameters(), lr=args.lr_a)
        else:
            alpha_opt = optim.Adam(generator.parameters(), lr=args.lr_a)
        opt_kit = [optimizer, alpha_opt]

    if args.dataset == "mnist":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    else:
        scheduler = "linear"

    model = model.to(args.device)
    model.train()
    return model, opt_kit, scheduler, start_epoch, generator


def train(dataset_kit, model_kit, args):
    train_l, train_ul, test_loader = dataset_kit
    model, optimizer, scheduler, start_epoch, generator = model_kit
    if 'inl0' in args.trainer or 'ax' in args.trainer:
        alpha_opt = optimizer[1]
        optimizer = optimizer[0]

    # Define losses.
    criterion = nn.CrossEntropyLoss()

    l0_ins = L0Layer(args)
    l0_ins.train()
    reg_component = IndL0VATOne(args)
    if "vat" in args.trainer:
        reg_component = VAT(args)

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
        log_alpha = generator(torch.FloatTensor(train_ul.data[idx]).to(args.device))
        show_and_save_generated_demo(l0_ins, log_alpha, train_ul.data[idx], args, 0, 0.1, 1, vis_set=vis_set)

    # train
    saved_count = 0
    start_time = time.time()
    decay_stop = False
    wlog("decay stop optimizing on generator %s" % str(decay_stop))
    for epoch in range(start_epoch, args.num_epochs):
        for it in range(args.num_batch_it):

            x, t = train_l.get(args.batch_size, aug_trans=args.aug_trans, aug_flip=args.aug_flip)
            images = torch.FloatTensor(x).to(args.device)
            labels = torch.LongTensor(t).to(args.device)

            x_u, t_for_debug = train_ul.get(args.ul_batch_size, aug_trans=args.aug_trans, aug_flip=args.aug_flip)
            ul_images = torch.FloatTensor(x_u).to(args.device)

            logits = model(images)

            # supervised loss
            ce_loss = criterion(logits, labels)
            sup_loss = ce_loss

            ul_loss = 0
            total_loss = sup_loss
            if "ce" == args.trainer:
                pass
            elif "inl0" in args.trainer:
                ul_loss, perturbations = reg_component(model, ul_images, generator, l0_ins, kl_way=args.kl)
                total_loss = sup_loss + ul_loss
                alpha_opt.zero_grad()
                dicts = {}
                if "l0" in args.trainer:
                    dif = (perturbations - ul_images).view(ul_images.shape[0], -1)
                    l2_dis = torch.norm(dif, p=2, dim=1)
                    mean, min, max = l2_dis.mean(), l2_dis.min(), l2_dis.max()
                    dicts['train/DisAvg'] = mean
                    dicts['train/DisMin'] = min
                    dicts['train/DisMax'] = max
                if args.vis:
                    vis_step(args.writer, epoch * args.num_batch_it + it, dicts)
            elif "ax" == args.trainer:
                ul_loss, perturbations = reg_component(model, ul_images, generator, l0_ins, alpha_opt, kl_way=args.kl)
                total_loss = sup_loss + ul_loss
            elif "axone" in args.trainer:
                ul_loss, perturbations = reg_component(model, ul_images, generator, l0_ins, kl_way=args.kl)
                total_loss = sup_loss + ul_loss
                alpha_opt.zero_grad()
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            total_loss.backward()

            if args.trainer in ["inl0", 'inl0ent', 'axone'] and (epoch < args.epoch_decay_start or (epoch > args.epoch_decay_start and not decay_stop)):
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
                        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': test_loss, 'acc': acc,
                                    'generator_state_dict': generator.conv.state_dict()},
                                   "%s/model.pth" % args.dir_path)
                        if epoch > args.num_epochs / 2:
                            shutil.copy("%s/model.pth" % args.dir_path, "%s/model_%d.pth" % (args.dir_path, epoch+1))
                    pred_y = torch.max(logits, dim=1)[1]
                    train_acc = 1.0 * torch.sum(pred_y == labels).item() / pred_y.shape[0]

                    dicts = {"Train/CELoss": ce_loss, "Train/UnsupLoss": ul_loss, "Train/Loss": total_loss, "Test/Acc": acc, "Test/Loss": test_loss, "Train/Acc": train_acc}
                    if "l0" in args.trainer:
                        dif = (perturbations - ul_images).view(ul_images.shape[0], -1)
                        l2_dis = torch.norm(dif, p=2, dim=1)
                        mean, min, max = l2_dis.mean(), l2_dis.min(), l2_dis.max()
                        dicts['Train/DisAvg'] = mean
                        dicts['Train/DisMin'] = min
                        dicts['Train/DisMax'] = max
                        np.savetxt("%s/demo/%d_dist.txt" % (args.dir_path, epoch), l2_dis.detach().cpu().numpy(), delimiter=',', fmt='%.2f', newline=',')
                    if args.vis:
                        vis_step(args.writer, epoch, dicts)

                    save_interval = 5
                    if args.epoch_decay_start + 10 > epoch >= args.epoch_decay_start:
                        save_interval = 1
                    if "l0" in args.trainer and epoch % save_interval == 0 and "fig" in args.log_arg:
                        log_alpha = generator(torch.FloatTensor(train_ul.data[idx]).to(args.device))
                        show_and_save_generated_demo(l0_ins, log_alpha, train_ul.data[idx], args, epoch+1, acc, saved_count, vis_set=vis_set)
                        saved_count += 1
                start_time = time.time()

        if scheduler == "linear":
            lr = adjust_learning_rate(optimizer, epoch, args)
            # if args.trainer in ["inl0", 'inl0ent', 'axone'] and args.dataset != 'svhn':
            #     adjust_learning_rate(alpha_opt, epoch, args, "decay" in args.log_arg)
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]
        if epoch % args.log_interval == 0:
            wlog("learning rate %f" % lr)
            if args.vis:
                args.writer.add_scalar("Optimizer/LearningRate", lr, epoch)
        show_weight_generator(generator, epoch, args)
        show_perturbations(train_ul.data, generator, l0_ins, epoch, args)


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
    except BaseException as err:
        traceback.print_exc()
        if arg.dir_path:
            shutil.rmtree(arg.dir_path)
