import torch
import numpy as np
import matplotlib.pyplot as plt


def show_image_mask(image, mask, r_mask, log_alpha, args, vis_image=None):
    if isinstance(image, torch.Tensor):
        if image.device.type == 'cuda':
            image = image.cpu()
        image = image.detach().numpy()

    if isinstance(mask, torch.Tensor):
        if mask.device.type == 'cuda':
            mask = mask.cpu()
        mask = mask.detach().numpy()

    if isinstance(r_mask, torch.Tensor):
        if r_mask.device.type == 'cuda':
            r_mask = r_mask.cpu()
        r_mask = r_mask.detach().numpy()

    if isinstance(log_alpha, torch.Tensor):
        if log_alpha.device.type == 'cuda':
            log_alpha = log_alpha.cpu()
        log_alpha = log_alpha.detach().numpy()

    eps = 1 if 'l0' not in args.trainer or args.eps < 1.001 else args.eps

    if args.dataset == 'mnist':
        image = image[0]
        masked_image = image * r_mask[0]
        if vis_image is not None:
            vis_image = vis_image[0]
    else:
        image = image.transpose((1, 2, 0))
        if vis_image is not None:
            vis_image = vis_image.transpose((1, 2, 0))
            masked_image = vis_image * r_mask.transpose((1, 2, 0))
        else:
            masked_image = image * r_mask.transpose((1, 2, 0))

    if (args.dataset in ['mnist', 'svhn'] and image.min() < 0.01) or args.data_dir == 'data0.5':
        image += 0.5

    mask = mask[0]
    r_mask = r_mask[0]
    log_alpha = log_alpha[0]

    fig, axes = plt.subplots(3, 2)

    plt.subplot(321)
    plt.imshow(image, cmap='gray' if args.dataset == 'mnist' else None)
    plt.subplot(322)
    plot1 = plt.imshow(log_alpha, cmap='gray')
    plt.subplot(323)
    plot2 = plt.imshow(mask, cmap='gray')
    plot2.set_clim(0.0, 1.0)
    plt.subplot(324)
    plot_r = plt.imshow(r_mask, cmap='gray')
    plot_r.set_clim(0.0, 1.0)

    if vis_image is not None:
        plt.subplot(325)
        plot_r = plt.imshow(vis_image)
        plot_r.set_clim(0.0, 1.0)
    plt.subplot(326)
    p_max, p_min = masked_image.max(), masked_image.min()
    # if args.dataset in ['masked_image - p_min']
    plot3 = plt.imshow(eps * masked_image, cmap='gray' if args.dataset == 'mnist' else None)
    if args.dataset in ['mnist', 'svhn']:
        plot3.set_clim(0, eps)
    # Create room on the right
    for ax in fig.axes:
        ax.axis('off')
    plt.gcf().subplots_adjust(right=0.7)
    # Make a new Axes instance
    cbar_ax = fig.add_axes([0.70, 0.15, 0.025, 0.7])
    fig.colorbar(plot1, cax=cbar_ax)
    cbar_ax = fig.add_axes([0.80, 0.15, 0.025, 0.7])
    fig.colorbar(plot2, cax=cbar_ax)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    fig.colorbar(plot3, cax=cbar_ax)
    return fig, mask, log_alpha


def show_and_save_generated_demo(l0_ins, log_alphas, images, args, epoch, acc, count, vis_set=None):
    l0_loss = l0_ins.l0_loss(log_alphas)
    r_masks = l0_ins.get_mask(log_alphas)
    r_masks = r_masks.cpu().detach().numpy()
    l0_ins.eval()
    masks = l0_ins.get_mask(log_alphas)
    masks = masks.cpu().detach().numpy()
    np.save('%s/demo/r_masks_%d.npy' % (args.dir_path, count), r_masks)
    np.save('%s/demo/log_alpha_%d.npy' % (args.dir_path, count), log_alphas.cpu().detach().numpy())
    for k, image in enumerate(images):
        fig, mask, log_alpha = show_image_mask(image, masks[k], r_masks[k], log_alphas[k], args, vis_image=vis_set[k])
        plt.suptitle('Epoch %d, Acc %.4g, l0 loss %.4g' % (epoch, acc, l0_loss.item()))
        fig.savefig('%s/demo/%d_%d.jpg' % (args.dir_path, k, count))
        plt.close()

    l0_ins.train()


def show_weight_generator(generator, epoch, args):
    for name, param in generator.named_parameters():
        args.writer.add_histogram("Model/" + name, param.clone().cpu().data.numpy(), epoch)


def show_perturbations(data_set, generator, l0_ins, epoch, args):
    dis_list = []
    mask_list = []
    l0_ins.eval()
    for i in range(0, 10000, 200):
        subset = torch.FloatTensor(data_set[i:i + 200]).to(args.device)
        log_alpha = generator(subset)
        masks = l0_ins.get_mask(log_alpha)
        perturbations = args.eps * subset * masks
        masks = masks.view(-1).cpu().data.numpy()
        dif = subset - perturbations
        dis = dif.view(dif.shape[0], -1).norm(p=2, dim=1).cpu().data.numpy()
        mask_list.append(masks)
        dis_list.append(dis)
    mask_list = np.concatenate(mask_list)
    dis_list = np.concatenate(dis_list)
    args.writer.add_histogram("Train/Mask", mask_list, epoch)
    args.writer.add_histogram("Train/Perturbation", dis_list, epoch)
