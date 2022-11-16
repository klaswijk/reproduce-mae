from collections import namedtuple

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DatasetInfo = namedtuple("DatasetInfo", ["image_size", "n_classes"])
info = {
    "cifar10": DatasetInfo(image_size=32, n_classes=10)
}


def cifar(train, batch_size, device, limit=-1):
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
    if limit > 0:
        dataset = Subset(dataset, range(limit))
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=str(device) != "cpu",
        pin_memory_device=str(device) if str(device) != "cpu" else ""
    )
    return dataloader


def get_dataloader(dataset, train, batch_size, device, limit=None):
    if dataset == "cifar10":
        return cifar(train, batch_size, device, limit)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")