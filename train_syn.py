import os
import argparse
import traceback

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_util
from six.moves import cPickle
import tensorboardX as tbX

from models.MLPSyn import MLPSyn
from torch_func.utils import set_framework_seed, weights_init_uniform
from Loss import L0Layer, L0VATOne, L0VAT, AutoL0Layer, L0VATSym
from visual_func import *


def parse_args():
    parser = argparse.ArgumentParser(description='xVAT Semi-supervised learning on synthetic datasets in PyTorch')
    parser.add_argument('--dataset', type=str, default='1', help='syndata-1, syndata-2 (default: 1)')
    parser.add_argument('--data-dir', type=str, default='data', help='default: data')
    parser.add_argument('--trainer', type=str, default='l0', help='ce, vat, l0 (default: l0)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='N', help='number of iterations (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default="", metavar='N', help='gpu id list (default: auto select)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='iterations to wait before logging status, (default: 10)')
    parser.add_argument('--ul-batch-size', type=int, default=1000, help='size of unlabeled data set (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-a', type=float, default=0.001, help='learning rate for log_alpha (default: 0.001)')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='learning rate (default: 0.995)')
    parser.add_argument('--eps', type=float, default=0.5, help='epsilon (default: 0.5)')
    parser.add_argument('--ent-min', action='store_true', default=False, help='visual by tensor board')

    parser.add_argument('--k', type=int, default=1, help='power iterations, (default: 1)')
    parser.add_argument('--lamb', type=float, default=1.0, help='lambda for unlabeled l0 loss (default: 1)')
    parser.add_argument('--lamb2', type=float, default=0.0, help='lambda for unlabeled smooth loss (default: 0)')
    parser.add_argument('--alpha', type=float, default=1, help='alpha for unlabeled loss of L0VAT (default: 1)')
    parser.add_argument('--zeta', type=float, default=1.1, help='zeta for L0VAT, always > 1 (default: 1.1)')
    parser.add_argument('--beta', type=float, default=0.66, help='beta for L0VAT (default: 0.66)')
    parser.add_argument('--gamma', type=float, default=-0.1, help='gamma for L0VAT, always < 0 (default: -0.1)')
    parser.add_argument('--kl', type=int, default=1, help='unlabel loss computing, (default: 1)')
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

    if "vat" in args.trainer:
        args.xi = 1e-6

    if not args.log_arg:
        args.log_arg = "trainer-lr"
        if "vat" in args.trainer:
            args.log_arg += "-eps-xi-kl"
        if "l0" in args.trainer:
            args.log_arg += "-lr_a-kl-lamb-eps"
        args.log_arg += "-seed"

    # use some parameters, pid and running time to mark the process
    if args.vis:
        args.dir_path = form_dir_path("L0VAT-semi", args)
        set_file_logger(logger, args)
        args.writer = tbX.SummaryWriter(log_dir=args.dir_path)

    wlog("args in this experiment:\n%s" % '\n'.join(str(e) for e in sorted(vars(args).items())))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    return args, kwargs


def get_data(args):
    with open('%s/syndata_%s.pkl' % (args.data_dir, args.dataset), "rb") as f:
        if sys.version_info.major == 3:
            dataset = cPickle.load(f, encoding='bytes')
        else:
            dataset = cPickle.load(f)

    x_train = torch.FloatTensor(np.asarray(dataset[0][0][0]))
    t_train = torch.LongTensor(np.asarray(dataset[0][0][1]))
    x_valid = torch.FloatTensor(np.asarray(dataset[0][1][0]))
    t_valid = torch.LongTensor(np.asarray(dataset[0][1][1]))

    train_loader = data_util.DataLoader(data_util.TensorDataset(x_train, t_train), 128, False)
    valid_loader = data_util.DataLoader(data_util.TensorDataset(x_valid, t_valid), 128, False)
    return x_train, t_train, x_valid, t_valid, train_loader, valid_loader, dataset


def init_model(args):
    model = MLPSyn(4 if args.dataset == "3" else 2)
    model.apply(weights_init_uniform)
    model = model.to(args.device)
    generator = None
    if 'ce' == args.trainer:
        optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.995)
    elif 'vat' in args.trainer:
        optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.995)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)
    elif 'l0' in args.trainer:
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)
        optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.995)
    else:
        raise NotImplementedError
    return model, optimizer, scheduler, generator


def train(dataset_kit, model_kit, components, args):
    x_train, t_train, x_valid, t_valid, train_loader, valid_loader, dataset = dataset_kit
    model, optimizer, scheduler, generator = model_kit

    err_rate_list = []
    criterion, reg_component, l0_ins, alpha_opt, log_alphas = components

    ul_i = 0
    loss = 0
    error_rate = 1.0
    local_best_rate = 100
    fig_count = 0

    idx = list(range(30))
    if args.vis:
        if "l0" in args.trainer:
            semi_name = "%s/pert_%d_00.jpg" % (args.dir_path, args.seed)
            log_alpha = log_alphas[idx]
            l0_ins.eval()
            masks = l0_ins.get_mask(log_alpha)
            m = args.eps * masks.cpu().detach().numpy() * x_valid[idx].cpu().numpy()
            visualize_adv_points(model, args.dataset, x_train, t_train, x_valid[idx], t_valid[idx], m, t_valid[idx], dataset[1], 0, 0, args, save_filename=semi_name)
            semi_name = "%s/pert_%d_%d.jpg" % (args.dir_path, args.seed, 0)
            l0_ins.train()
            masks = l0_ins.get_mask(log_alpha)
            m = args.eps * masks.cpu().detach().numpy() * x_valid[idx].cpu().numpy()
            visualize_adv_points(model, args.dataset, x_train, t_train, x_valid[idx], t_valid[idx], m, t_valid[idx], dataset[1], 0, 0, args, save_filename=semi_name)
        if "vat" in args.trainer:
            _, m = reg_component(model, x_valid[idx].to(args.device), return_adv=True)
            m = (m.cpu() + x_valid[idx]).numpy()
            semi_name = "%s/pert_%d_%d.jpg" % (args.dir_path, args.seed, fig_count)
            visualize_adv_points(model, args.dataset, x_train, t_train, x_valid[idx], t_valid[idx], m, t_valid[idx], dataset[1], 0, 0, args, save_filename=semi_name)
        semi_name = "%s/step_%d_%d.jpg" % (args.dir_path, args.seed, fig_count)
        visualize_contour_semi(model, args.dataset, x_train, t_train, x_valid, t_valid, dataset[1], valid_loader, args, save_filename=semi_name)

    if "l0" in args.trainer:
        idx = list(range(10))
        log_alpha = log_alphas[idx]
        masks = l0_ins.get_mask(log_alpha)
        wlog("avg log alpha is %g, shape %s" % (torch.mean(log_alpha), str(log_alpha.shape)))
        wlog("avg mask is %g, shape %s" % (torch.mean(masks), str(masks.shape)))

    fig_count += 1
    for i in range(1, args.iterations+1):
        ce_loss, ul_loss = 0, 0
        for l_x, l_y in train_loader:
            l_x = l_x.to(args.device)
            l_y = l_y.to(args.device)
            ul_x = torch.FloatTensor(x_valid[ul_i * args.ul_batch_size:ul_i * args.ul_batch_size + args.ul_batch_size]).to(args.device)
            index = np.arange(ul_i * args.ul_batch_size, ul_i * args.ul_batch_size + args.ul_batch_size)
            ul_i = 0 if ul_i >= x_valid.shape[0] / args.ul_batch_size - 1 else ul_i + 1
            logits = model(l_x)
            ce_loss = criterion(logits, l_y)
            if "ce" == args.trainer:
                loss = ce_loss
                ul_loss = 0
            elif args.trainer == "vat":
                ul_loss = reg_component(model, l_x)
                loss = ce_loss + ul_loss
            elif "ce-vat" in args.trainer:
                ul_loss = reg_component(model, ul_x)
                loss = ce_loss + ul_loss
            elif "ce-l0" in args.trainer:
                ul_loss, mask = reg_component(model, ul_x, log_alphas, l0_ins, kl_way=args.kl)
                loss = ce_loss + args.alpha * ul_loss
                alpha_opt.zero_grad()
            elif args.trainer == "ce-l02":
                ul_loss, mask = reg_component(model, ul_x, log_alphas, index, l0_ins, alpha_opt, kl_way=args.kl)
                loss = ce_loss + args.alpha * ul_loss
            else:
                raise NotImplementedError
            optimizer.zero_grad()
            loss.backward()
            if "l0" in args.trainer and i < 5000:
                for p in alpha_opt.param_groups[0]['params']:
                    if p.grad is None:
                        continue
                    p.grad.data = -p.grad.data
                alpha_opt.step()

            optimizer.step()
            scheduler.step()

        if i % args.log_interval == 0:
            lr = scheduler.get_lr()[-1]
            test_err, test_loss = evaluate_classifier(model, valid_loader, args.device)
            error_rate = 1.0 * test_err / x_valid.shape[0]
            wlog("iteration %d, train loss %g, ce %g, ul %g, test error rate %g and test loss %g" % (i, loss, ce_loss, ul_loss, error_rate, test_loss))
            wlog("lr %g" % lr)
            if args.vis:
                dicts = {"Train/CELoss": ce_loss, "Train/UnsupLoss": ul_loss, "Train/Loss": loss, "Test/ErrorRate": error_rate, "Test/Loss": test_loss,
                         "Optimizer/LearningRate": lr}
                if "l0a" in args.trainer:
                    dicts["Test/eps"] = l0_ins.eps.max().item()
                vis_step(args.writer, i, dicts)
            err_rate_list.append(error_rate)
            if error_rate < local_best_rate:
                local_best_rate = error_rate

            idx = list(range(30))
            if "vat" in args.trainer and args.vis:
                _, m = reg_component(model, x_valid[idx].to(args.device), return_adv=True)
                m = (m.cpu() + x_valid[idx]).numpy()
                if args.vis and i % args.log_interval == 0:
                    semi_name = "%s/pert_%d_%d.jpg" % (args.dir_path, args.seed, fig_count)
                    visualize_adv_points(model, args.dataset, x_train, t_train, x_valid[idx], t_valid[idx], m, t_valid[idx], dataset[1], i, test_err, args, save_filename=semi_name)

            if "l0" in args.trainer:
                log_alpha = log_alphas[idx]
                masks = l0_ins.get_mask(log_alpha)
                wlog("avg log alpha is %g, shape %s" % (torch.mean(log_alpha), str(log_alpha.shape)))
                wlog("avg mask is %g, shape %s" % (torch.mean(masks), str(masks.shape)))
                m = args.eps * masks.cpu().detach().numpy() * x_valid[idx].cpu().numpy()
                if args.vis and i % args.log_interval == 0:
                    semi_name = "%s/pert_%d_%d.jpg" % (args.dir_path, args.seed, fig_count)
                    visualize_adv_points(model, args.dataset, x_train, t_train, x_valid[idx], t_valid[idx], m, t_valid[idx], dataset[1], i, test_err, args, save_filename=semi_name)
                if "l0a" in args.trainer:
                    wlog("eps %g" % l0_ins.eps)
        if args.vis and i % args.log_interval == 0:
            semi_name = "%s/step_%d_%d.jpg" % (args.dir_path, args.seed, fig_count)
            visualize_contour_semi(model, args.dataset, x_train, t_train, x_valid, t_valid, dataset[1], valid_loader, args, save_filename=semi_name)
            fig_count += 1
    return err_rate_list, error_rate, local_best_rate


def main(args):
    set_framework_seed(args.seed, args.debug)

    l0_ins = L0Layer(args)
    if args.trainer == "ce-l0a":
        l0_ins = AutoL0Layer(args)
    l0_ins.train()

    args.xi = 1e-6
    reg_component = None
    alpha_opt = None
    log_alpha = None
    if "vat" in args.trainer:
        reg_component = VAT(args)
    elif args.trainer == "ce-l0":
        log_alpha = torch.randn((1000, 100), device=args.device)
        log_alpha.requires_grad = True
        alpha_opt = optim.Adam([log_alpha], lr=args.lr_a)
        reg_component = L0VATOne(args)
    elif args.trainer == "ce-l02":
        log_alpha = torch.randn((1000, 100), device=args.device)
        log_alpha.requires_grad = True
        reg_component = L0VAT(args)
        alpha_opt = optim.Adam([log_alpha], lr=args.lr_a)

    # wlog("log alpha avg %g" % log_alpha.mean())
    criterion = nn.CrossEntropyLoss()
    components = [criterion, reg_component, l0_ins, alpha_opt, log_alpha]

    total_exp = 1
    best_err_rate = 100
    best_model = None
    avg_error = 0
    avg_local_error = 0

    for exp in range(total_exp):
        seed = exp + args.seed
        set_framework_seed(seed, args.debug)

        dataset_kit = get_data(args)
        model_kit = init_model(args)
        model, optimizer, scheduler, generator = model_kit

        error_list, error_rate, local_best_rate = train(dataset_kit, model_kit, components, args)

        x_train, t_train, x_valid, t_valid, train_loader, valid_loader, dataset = dataset_kit

        avg_error += error_rate
        avg_local_error += local_best_rate
        if args.vis:
            semi_name = "%s/%d_final.jpg" % (args.dir_path, exp)
            visualize_contour_semi(model, args.dataset, x_train, t_train, x_valid, t_valid, dataset[1], valid_loader, args, save_filename=semi_name)

        if error_rate < best_err_rate:
            best_err_rate = error_rate
            best_model = model

    avg_error /= total_exp
    avg_local_error /= total_exp
    wlog("avg error rate %g" % avg_error)
    wlog("avg local best error rate %g" % avg_local_error)
    wlog("best error rate %g" % best_err_rate)
    if args.vis:
        np.save("%s/error_rate.txt" % args.dir_path, avg_error)
        torch.save(best_model.state_dict(), "%s/syndata-%s-model-%g-%g.pkl" % (args.dir_path, args.dataset, best_err_rate, avg_error))


if __name__ == '__main__':
    arg, _ = parse_args()
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
