import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _cifar10_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, test_tf


def build_cifar10_loaders(cfg: dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    root = data_cfg.get("root", "./data")
    batch_size = train_cfg["batch_size"]
    num_workers = int(data_cfg.get("num_workers", 0))

    train_tf, test_tf = _cifar10_transforms()

    train_set = datasets.CIFAR10(
        root, train=True, download=True, transform=train_tf
    )
    test_set = datasets.CIFAR10(
        root, train=False, download=True, transform=test_tf
    )

    n_train = data_cfg.get("subset_train")
    if n_train is not None:
        n_train = int(n_train)
        indices = list(range(min(n_train, len(train_set))))
        train_set = Subset(train_set, indices)

    n_test = data_cfg.get("subset_test")
    if n_test is not None:
        n_test = int(n_test)
        indices = list(range(min(n_test, len(test_set))))
        test_set = Subset(test_set, indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
