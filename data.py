from collections import namedtuple

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DatasetInfo = namedtuple("DatasetInfo", ["image_size", "n_classes"])
info = {
    "cifar10": DatasetInfo(image_size=32, n_classes=10)
}


def cifar(train, device, checkpoint):
    """Returns (dataloader, image_size)"""
    data_path = checkpoint["data_path"]
    limit = checkpoint["config"]["data"]["limit"]
    val_ratio = checkpoint["config"]["data"]["val_ratio"]
    batch_size = checkpoint["config"]["batch_size"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root=data_path,
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


def get_dataloader(dataset, train, device, checkpoint):
    if dataset == "cifar10":

        return cifar(train, device, checkpoint)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
