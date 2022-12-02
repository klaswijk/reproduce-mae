import os
import json
from collections import namedtuple
from pandas import read_csv
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

DatasetInfo = namedtuple(
    "DatasetInfo", ["image_size", "n_classes", "multilabel"])
info = {
    "cifar10": DatasetInfo(image_size=32, n_classes=10, multilabel=False),
    "imagenette": DatasetInfo(image_size=160, n_classes=10, multilabel=False),
    "coco": DatasetInfo(image_size=64, n_classes=80, multilabel=True)
}


class ImageNetteDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None, test=False):
        # Should have 10000 train and 3395 val ("test") images
        self.img_labels = read_csv(path + "noisy_imagenette.csv")
        if test:
            self.img_labels = self.img_labels[10000:]
        else:
            self.img_labels = self.img_labels[:10000]
        self.img_dir = path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CocoMultilabel(Dataset):

    def __init__(self, path, transform=None, target_transform=None, test=False, version="2017"):
        datatype = "val" if test else "train"
        with open(f"{path}/coco/annotations/instances_{datatype}{version}.json", "r") as f:
            instances = json.load(f)

        self.transform = transform
        self.target_transform = target_transform
        self.datapath = f"{path}/coco/{datatype}{version}"
        self.images = [image["file_name"] for image in instances["images"]]
        self.image_ids = [image["id"] for image in instances["images"]]
        self.label_names = [cat["name"] for cat in instances["categories"]]
        self.labels = {id: np.zeros(80) for id in self.image_ids}

        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        ignore = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]

        for annotation in instances["annotations"]:
            cat_id = annotation["category_id"]
            cat_id -= np.searchsorted(ignore, cat_id)
            image_id = annotation["image_id"]
            self.labels[image_id][cat_id - 1] = 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(f"{self.datapath}/{self.images[idx]}")
        image = image.convert('RGB')
        label = self.labels[self.image_ids[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def coco(train, device, checkpoint, transform):
    """Returns (dataloader, image_size)"""
    data_path = checkpoint["data_path"]
    limit = checkpoint["config"]["data"]["limit"]
    val_ratio = checkpoint["config"]["data"]["val_ratio"]
    batch_size = checkpoint["config"]["batch_size"]

    if train:
        dataset = CocoMultilabel(
            data_path,
            transform=transform,
        )

        idx = np.array(list(range(len(dataset))))
        np.random.shuffle(idx)

        if limit and limit > -1:
            dataset = Subset(dataset, idx[: limit])
            idx = np.array(list(range(len(dataset))))
            np.random.shuffle(idx)

        valsize = int(val_ratio * len(dataset))
        valset = Subset(dataset, idx[-valsize:])
        trainset = Subset(dataset, idx[: -valsize])
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
        dataset = CocoMultilabel(
            data_path,
            transform=transform,
            test=True
        )
        testloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=str(device) != "cpu",
            pin_memory_device=str(device) if str(device) != "cpu" else ""
        )
        return testloader


def imagenette(train, device, checkpoint):
    """Returns (dataloader, image_size)"""
    data_path = checkpoint["data_path"]
    limit = checkpoint["config"]["data"]["limit"]
    val_ratio = checkpoint["config"]["data"]["val_ratio"]
    batch_size = checkpoint["config"]["batch_size"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop((160, 160)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    targets = {
        "n01440764": 0,
        "n02979186": 1,
        "n03028079": 2,
        "n03417042": 3,
        "n03445777": 4,
        "n02102040": 5,
        "n03000684": 6,
        "n03394916": 7,
        "n03425413": 8,
        "n03888257": 9,
    }

    target_transform = transforms.Lambda(targets.get)

    if train:
        dataset = ImageNetteDataset(
            data_path + "/imagenette2-160/",
            transform=transform,
            target_transform=target_transform
        )

        idx = np.array(list(range(len(dataset))))
        np.random.shuffle(idx)

        if limit and limit > -1:
            dataset = Subset(dataset, idx[: limit])
            idx = np.array(list(range(len(dataset))))
            np.random.shuffle(idx)

        valsize = int(val_ratio * len(dataset))
        valset = Subset(dataset, idx[-valsize:])
        trainset = Subset(dataset, idx[: -valsize])
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
        dataset = ImageNetteDataset(
            data_path + "/imagenette2-160/",
            transform=transform,
            target_transform=target_transform,
            test=True
        )
        testloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=str(device) != "cpu",
            pin_memory_device=str(device) if str(device) != "cpu" else ""
        )
        return testloader


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

    if train:
        idx = np.array(list(range(len(dataset))))
        np.random.shuffle(idx)

        valsize = int(val_ratio * len(dataset))
        valset = Subset(dataset, idx[-valsize:])
        trainset = Subset(dataset, idx[: -valsize])
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


def get_dataloader(dataset, train, device, checkpoint, transform_type=None):
    if not train:
        transform_type = None
    if transform_type == "finetune":
        if dataset == "coco":
            transform = transforms.Compose([
                transforms.Resize(80, antialias=True),
                transforms.RandAugment(),
                transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.RandAugment(),
                transforms.RandomCrop(info[dataset].image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    elif transform_type == "pretrain":
        if dataset == "coco":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(64, ratio=(1, 1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomCrop(info[dataset].image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(info[dataset].image_size, antialias=True),
            transforms.RandomCrop(info[dataset].image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if dataset == "cifar10":
        return cifar(train, device, checkpoint)
    elif dataset == "imagenette":
        return imagenette(train, device, checkpoint)
    elif dataset == "coco":
        return coco(train, device, checkpoint, transform)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
