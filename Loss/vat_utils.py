import torch
import numpy as np
import matplotlib.pyplot as plt


def show_image_adv(image, r_adv, args):
    if isinstance(image, torch.Tensor):
        if image.device.type == "cuda":
            image = image.cpu().detach()
        image = image.numpy()

    if isinstance(r_adv, torch.Tensor):
        if r_adv.device.type == "cuda":
            r_adv = r_adv.cpu().detach()
        r_adv = r_adv.numpy()

    if args.dataset == "mnist":
        image = image[0]
        example_image = image + r_adv[0]
    elif args.dataset == "svhn":
        image = image.transpose((1, 2, 0))
        if args.data_dir != "data1.0":
            image += 0.5
        example_image = image + r_adv.transpose((1, 2, 0))
    else:
        if args.data_dir == "data0.5":
            image += 0.5
        image = image.transpose((1, 2, 0))
        example_image = image + r_adv.transpose((1, 2, 0))

    r_adv = r_adv[0]

    figure = plt.figure()
    # Create room on the right
    plt.gcf().subplots_adjust(right=0.8)

    plt.subplot(131)
    plt.imshow(image, cmap='gray' if args.dataset == "mnist" else None)
    plt.subplot(132)
    plt.imshow(example_image, cmap='gray' if args.dataset == "mnist" else None)
    plt.subplot(133)
    plot = plt.imshow(r_adv)
    # Make a new Axes instance
    c_bar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
    figure.colorbar(plot, cax=c_bar_ax)
    return figure, r_adv


def show_and_save_vat_generated_demo(ins, model, images, args, epoch, acc, save_interval):
    vat_loss, r_adv = ins(model, images, return_adv=True)
    step = epoch
    if epoch != 0:
        step = int(epoch / save_interval)
    r_adv = r_adv.cpu().detach().numpy()
    np.save("%s/demo/r_adv_%d.npy" % (args.dir_path, step), r_adv)
    for k, image in enumerate(images):
        fig, mask = show_image_adv(image, r_adv[k], args)
        plt.suptitle("Epoch %d, Acc %.4g, l0 loss %.4g" % (epoch, acc, vat_loss.item()))
        fig.savefig("%s/demo/%d_%d.jpg" % (args.dir_path, k, step))
        plt.close()
        plt.close()


def show_and_save_at_generated_demo(ins, model, images, labels, args, epoch, acc, save_interval):
    if isinstance(images, np.ndarray):
        images = torch.FloatTensor(images).to(args.device)
        labels = torch.LongTensor(labels).to(args.device)
    vat_loss, r_adv = ins(model, images, labels, return_adv=True)
    step = epoch
    if epoch != 0:
        step = int(epoch / save_interval)
    r_adv = r_adv.cpu().detach().numpy()
    np.save("%s/demo/r_adv_%d.npy" % (args.dir_path, step), r_adv)
    for k, image in enumerate(images):
        fig, mask = show_image_adv(image, r_adv[k], args)
        plt.suptitle("Epoch %d, Acc %.4g, l0 loss %.4g" % (epoch, acc, vat_loss.item()))
        fig.savefig("%s/demo/%d_%d.jpg" % (args.dir_path, k, step))
        plt.close()
        plt.close()
