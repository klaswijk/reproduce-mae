from collections import namedtuple

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DatasetInfo = namedtuple("DatasetInfo", ["image_size", "n_classes"])
info = {
    "cifar10": DatasetInfo(image_size=32, n_classes=10)
}


def cifar(train, batch_size, device, limit=-1, val_ratio=0.1):
    """Returns (dataloader, image_size)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    if limit and limit > -1:
        dataset = Subset(dataset, range(limit))

    idx = list(range(len(dataset)))
    if train:
        valsize = int(val_ratio * len(dataset))
        valset = Subset(dataset, idx[-valsize:])
        trainset = Subset(dataset, idx[:-valsize])
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=str(device) != "cpu",
            pin_memory_device=str(device) if str(device) != "cpu" else ""
        )
        valloader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=str(device) != "cpu",
            pin_memory_device=str(device) if str(device) != "cpu" else ""
        )
        return trainloader, valloader
    else:
        testloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=str(device) != "cpu",
            pin_memory_device=str(device) if str(device) != "cpu" else ""
        )
        return testloader


def get_dataloader(dataset, train, batch_size, device, limit=None, val_ratio=0.1):
    if dataset == "cifar10":
        return cifar(train, batch_size, device, limit, val_ratio)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
