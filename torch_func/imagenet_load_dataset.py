import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def relabel_dataset(dataset, n_l=13000):
    labeled_idx = []
    s_c = int(n_l / 1000)
    labels = np.array(dataset.targets)
    np.random.shuffle(labels)
    for i in range(1000):
        ind = np.where(labels == i)[0]
        labeled_idx.append(np.random.choice(ind, s_c))
    labeled_idx = np.concatenate(labeled_idx)
    # unlabeled_idx = np.delete(np.arange(len(labels)), labeled_idx)
    unlabeled_idx = np.arange(len(labels))

    return labeled_idx, unlabeled_idx


def load_dataset(data_dir, args):
    if "64" not in data_dir:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.482, 0.458, 0.408],
                                         std=[0.269, 0.261, 0.276])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    if "image64" not in data_dir:
        train_transforms = [transforms.RandomResizedCrop(224)] + train_transforms
    test_transforms = [transforms.ToTensor(), normalize]
    if "image64" not in data_dir:
        test_transforms = [transforms.Resize(256), transforms.CenterCrop(224)] + test_transforms
    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose(train_transforms))

    labeled_idx, unlabeled_idx = relabel_dataset(train_dataset, args.size)

    val_set = datasets.ImageFolder(val_dir, transforms.Compose(test_transforms))

    return train_dataset, labeled_idx, unlabeled_idx, val_set
