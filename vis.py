import math
import matplotlib.pyplot as plt
import numpy as np
import os


def vis_conv_filters(weight, bias, epoch, dir_name, val_accuracy, name, save=True):

    fig_size = (10, 10)
    # if c.shape[1] != c.shape[2]:
    #     fig_size = (20, 20)
    t = 8
    fig, axes = plt.subplots(nrows=t, ncols=t, figsize=fig_size)

    # cmap.set_under('white')
    i = 0
    c = weight - np.min(weight)
    c = c / np.max(c)
    l, m, h = np.min(weight), np.mean(weight), np.max(weight)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(c[i])
        if c[i].max() < 1.0 and c[i].min() > 0:
            im.set_clim(0.0, 1.0)
        i += 1

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.03, right=0.83, wspace=0.15, hspace=0.15)

    cb_ax = fig.add_axes([0.86, 0.1, 0.05, 0.8])
    fig.colorbar(im, cax=cb_ax)

    plt.title('Epoch: %d  Accuracy: %g, l:%g, m:%g, h:%g' % (epoch, val_accuracy, l, m, h), x=-9)

    if not os.path.exists('fig/{}'.format(dir_name)):
        os.makedirs('fig/{}/conv'.format(dir_name))
        os.makedirs('fig/{}/fc'.format(dir_name))

    if save:
        plt.savefig('fig/{}/conv/{}.png'.format(dir_name, name))
    else:
        plt.show()
    plt.close(fig)

    plt.hist(weight.reshape(-1), bins=50)
    plt.show()
    if save:
        plt.savefig('fig/{}/conv/hist_weight_{}.png'.format(dir_name, name))
    plt.close()

    plt.hist(bias.reshape(-1), bins=50)
    plt.show()
    if save:
        plt.savefig('fig/{}/conv/hist_bias_{}.png'.format(dir_name, name))
    plt.close()
