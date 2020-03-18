import torch.nn.functional as nfunc
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as plot_patch

from ExpUtils import *
from torch_func.evaluate import evaluate_classifier
from Loss import VAT


def visualize_contour_semi(model, d_i, x_data, y_data, ul_x, ul_y, basis, val_loader, args, with_lds=True, save_filename='prob_cont', show_contour=True):
    line_width = 10

    range_x = np.arange(-2.0, 2.1, 0.05)
    a_inv = linalg.inv(np.dot(basis, basis.T))
    train_x_org = np.dot(x_data, np.dot(basis.T, a_inv))
    test_x_org = np.zeros((range_x.shape[0] ** 2, 2))
    train_x_1_ind = np.where(y_data == 1)[0]
    train_x_0_ind = np.where(y_data == 0)[0]

    ul_x_org = np.dot(ul_x, np.dot(basis.T, a_inv))
    ul_x_1_ind = np.where(ul_y == 1)[0]
    ul_x_0_ind = np.where(ul_y == 0)[0]

    for i in range(range_x.shape[0]):
        for j in range(range_x.shape[0]):
            test_x_org[range_x.shape[0] * i + j, 0] = range_x[i]
            test_x_org[range_x.shape[0] * i + j, 1] = range_x[j]

    test_x = np.dot(test_x_org, basis)
    model.eval()
    f_p_y_given_x = model(torch.FloatTensor(test_x).to(args.device))
    pred = nfunc.softmax(f_p_y_given_x, dim=1)[:, 1].cpu().detach().numpy()

    z = np.zeros((range_x.shape[0], range_x.shape[0]))
    for i in range(range_x.shape[0]):
        for j in range(range_x.shape[0]):
            z[i, j] = pred[range_x.shape[0] * i + j]

    y, x = np.meshgrid(range_x, range_x)

    font_size = 20
    rc = 'r'
    bc = 'b'

    if d_i == "1":
        rescale = 1.0  # /np.sqrt(500)
        arc1 = plot_patch.Arc(xy=(0.5 * rescale, -0.25 * rescale), width=2.0 * rescale, height=2.0 * rescale, angle=0, theta1=270,
                              theta2=180, linewidth=line_width, alpha=0.15, color=rc)
        arc2 = plot_patch.Arc(xy=(-0.5 * rescale, +0.25 * rescale), width=2.0 * rescale, height=2.0 * rescale, angle=0, theta1=90,
                              theta2=360, linewidth=line_width, alpha=0.15, color=bc)
        fig = plt.gcf()
        frame = fig.gca()
        frame.add_artist(arc1)
        frame.add_artist(arc2)
    else:
        rescale = 1.0  # /np.sqrt(500)
        circle1 = plot_patch.Circle((0, 0), 1.0 * rescale, color=rc, alpha=0.2, fill=False, linewidth=line_width)
        circle2 = plot_patch.Circle((0, 0), 0.15 * rescale, color=bc, alpha=0.2, fill=False, linewidth=line_width)
        fig = plt.gcf()
        frame = fig.gca()
        frame.add_artist(circle1)
        frame.add_artist(circle2)

    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.scatter(ul_x_org[ul_x_1_ind, 0] * rescale, ul_x_org[ul_x_1_ind, 1] * rescale, s=2, marker='o', c=rc, label='$y=1$')
    plt.scatter(ul_x_org[ul_x_0_ind, 0] * rescale, ul_x_org[ul_x_0_ind, 1] * rescale, s=2, marker='o', c=bc, label='$y=0$')
    plt.scatter(train_x_org[train_x_1_ind, 0] * rescale, train_x_org[train_x_1_ind, 1] * rescale, s=50, marker='o', c=rc, label='$y=1$',
                edgecolor='black', linewidth=1)
    plt.scatter(train_x_org[train_x_0_ind, 0] * rescale, train_x_org[train_x_0_ind, 1] * rescale, s=50, marker='o', c=bc, label='$y=0$',
                edgecolor='black', linewidth=1)

    err_num, loss = evaluate_classifier(model, val_loader, args.device)
    err_rate = 1.0 * err_num / len(val_loader.dataset)

    lds_part = ""
    if with_lds:
        eps = args.eps
        args.eps = 0.5
        args.k = 5
        reg_component = VAT(args)
        x_data = x_data.to(args.device)

        ave_lds = 0
        for t in range(20):
            ave_lds += reg_component(model, x_data, kl_way=1)
        ave_lds /= 20
        lds_part = ' $\widetilde{\\rm LDS}=%.3f$' % ave_lds
        args.k = 1
        args.eps = eps
    fig.set_size_inches(5, 5)
    if save_filename is None:
        plt.show(block=False)
    else:
        cs = None
        if show_contour:
            levels = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
            cs = plt.contour(x * rescale, y * rescale, z, 7, cmap='bwr', vmin=0., vmax=1.0, linewidths=8., levels=levels)
            plt.setp(cs.collections, linewidth=1.0)
            plt.contour(x * rescale, y * rescale, z, 1, cmap='binary', vmin=0, vmax=0.5, linewidths=2.0)
        plt.tight_layout()
        plt.savefig(save_filename)
        if show_contour:
            plt.title('%s\nError %g%s' % (args.exp_marker, err_rate, lds_part))
        fig.set_size_inches(8, 6)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlim([-2. * rescale, 2. * rescale])
        plt.ylim([-2. * rescale, 2. * rescale])
        plt.xticks([-2.0, -1.0, 0, 1, 2.0], fontsize=font_size)
        plt.yticks([-2.0, -1.0, 0, 1, 2.0], fontsize=font_size)
        if show_contour and cs is not None:
            color_bar = plt.colorbar(cs)
            color_bar.ax.tick_params(labelsize=font_size)
        plt.savefig(save_filename.replace("step", "title"))
    plt.close()


def visualize_adv_points(model, d_i, x_data, y_data, ul_x, ul_y, adv_x, adv_y, basis, it, rate, args, save_filename='prob_cont', show_contour=True):
    line_width = 10

    range_x = np.arange(-2.0, 2.1, 0.05)
    a_inv = linalg.inv(np.dot(basis, basis.T))
    train_x_org = np.dot(x_data, np.dot(basis.T, a_inv))
    test_x_org = np.zeros((range_x.shape[0] ** 2, 2))
    train_x_1_ind = np.where(y_data == 1)[0]
    train_x_0_ind = np.where(y_data == 0)[0]

    ul_x_org = np.dot(ul_x, np.dot(basis.T, a_inv))
    ul_x_1_ind = np.where(ul_y == 1)[0]
    ul_x_0_ind = np.where(ul_y == 0)[0]

    adv_x_org = np.dot(adv_x, np.dot(basis.T, a_inv))
    adv_x_1_ind = np.where(adv_y == 1)[0]
    adv_x_0_ind = np.where(adv_y == 0)[0]

    for i in range(range_x.shape[0]):
        for j in range(range_x.shape[0]):
            test_x_org[range_x.shape[0] * i + j, 0] = range_x[i]
            test_x_org[range_x.shape[0] * i + j, 1] = range_x[j]

    test_x = np.dot(test_x_org, basis)
    model.eval()
    f_p_y_given_x = model(torch.FloatTensor(test_x).to(args.device))
    pred = nfunc.softmax(f_p_y_given_x, dim=1)[:, 1].cpu().detach().numpy()

    z = np.zeros((range_x.shape[0], range_x.shape[0]))
    for i in range(range_x.shape[0]):
        for j in range(range_x.shape[0]):
            z[i, j] = pred[range_x.shape[0] * i + j]

    y, x = np.meshgrid(range_x, range_x)

    font_size = 20
    rc = 'r'
    bc = 'b'

    gc = 'g'
    grc = 'gray'

    if d_i == "1":
        rescale = 1.0  # /np.sqrt(500)
        arc1 = plot_patch.Arc(xy=(0.5 * rescale, -0.25 * rescale), width=2.0 * rescale, height=2.0 * rescale, angle=0, theta1=270,
                              theta2=180, linewidth=line_width, alpha=0.15, color=rc)
        arc2 = plot_patch.Arc(xy=(-0.5 * rescale, +0.25 * rescale), width=2.0 * rescale, height=2.0 * rescale, angle=0, theta1=90,
                              theta2=360, linewidth=line_width, alpha=0.15, color=bc)
        fig = plt.gcf()
        frame = fig.gca()
        frame.add_artist(arc1)
        frame.add_artist(arc2)
    else:
        rescale = 1.0  # /np.sqrt(500)
        circle1 = plot_patch.Circle((0, 0), 1.0 * rescale, color=rc, alpha=0.2, fill=False, linewidth=line_width)
        circle2 = plot_patch.Circle((0, 0), 0.15 * rescale, color=bc, alpha=0.2, fill=False, linewidth=line_width)
        fig = plt.gcf()
        frame = fig.gca()
        frame.add_artist(circle1)
        frame.add_artist(circle2)

    plt.scatter(adv_x_org[adv_x_1_ind, 0] * rescale, adv_x_org[adv_x_1_ind, 1] * rescale, s=25, marker='o', c=rc, label='$y=1$')
    plt.scatter(adv_x_org[adv_x_0_ind, 0] * rescale, adv_x_org[adv_x_0_ind, 1] * rescale, s=25, marker='o', c=bc, label='$y=0$')
    plt.scatter(ul_x_org[ul_x_1_ind, 0] * rescale, ul_x_org[ul_x_1_ind, 1] * rescale, s=10, marker='o', c=rc, label='$y=1$')
    plt.scatter(ul_x_org[ul_x_0_ind, 0] * rescale, ul_x_org[ul_x_0_ind, 1] * rescale, s=10, marker='o', c=bc, label='$y=0$')

    plt.scatter(train_x_org[train_x_1_ind, 0] * rescale, train_x_org[train_x_1_ind, 1] * rescale, s=50, marker='o', c=gc, label='$y=1$',
                edgecolor='black', linewidth=1)
    plt.scatter(train_x_org[train_x_0_ind, 0] * rescale, train_x_org[train_x_0_ind, 1] * rescale, s=50, marker='o', c=grc, label='$y=0$',
                edgecolor='black', linewidth=1)

    fig.set_size_inches(5, 5)
    if save_filename is None:
        plt.show(block=False)
    else:
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        cs = plt.contour(x * rescale, y * rescale, z, 7, cmap='bwr', vmin=0., vmax=1.0, linewidths=8., levels=levels)
        plt.setp(cs.collections, linewidth=1.0)
        plt.contour(x * rescale, y * rescale, z, 1, cmap='binary', vmin=0, vmax=0.5, linewidths=2.0)
        plt.title('Iteration %d, error rate %g' % (it, rate))
        fig.set_size_inches(8, 6)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlim([-2. * rescale, 2. * rescale])
        plt.ylim([-2. * rescale, 2. * rescale])
        plt.xticks([-2.0, -1.0, 0, 1, 2.0], fontsize=font_size)
        plt.yticks([-2.0, -1.0, 0, 1, 2.0], fontsize=font_size)
        # if show_contour and cs is not None:
        #     color_bar = plt.colorbar(cs)
        #     color_bar.ax.tick_params(labelsize=font_size)
        plt.savefig(save_filename)
    plt.close()
