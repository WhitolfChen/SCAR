import torch
from torchvision import datasets, transforms

def get_transform(dataset_name):
    if dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    elif dataset_name == "imagenet50":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        raise ValueError("Dataset {} not found!".format(dataset_name))

    return train_transform, test_transform


def get_dataset(dataset_name):
    train_transform, test_transform = get_transform(dataset_name)
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=r"./data/cifar10", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=r"./data/cifar10", train=False, download=True, transform=test_transform
        )
    elif dataset_name == "imagenet50":
        train_dataset = datasets.ImageFolder(
            root=r"./data/imagenet50/train",
            transform=train_transform,
        )
        test_dataset = datasets.ImageFolder(
            root=r"./data/imagenet50/test",
            transform=test_transform,
        )
    else:
        raise ValueError("Dataset {} not found!".format(dataset_name))
    return train_dataset, test_dataset

def get_dataloader(dataset_name, batch_size):
    train_dataset, test_dataset = get_dataset(dataset_name)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    return train_dataloader, test_dataloader