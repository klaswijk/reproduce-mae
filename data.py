import os
from collections import namedtuple
from pandas import read_csv
from torch.utils.data import DataLoader, Subset, Dataset
from pycocotools.coco import COCO
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

DatasetInfo = namedtuple(
    "DatasetInfo", ["image_size", "n_classes", "multilabel"])
info = {
    "cifar10": DatasetInfo(image_size=32, n_classes=10, multilabel=False),
    "imagenette": DatasetInfo(image_size=160, n_classes=10, multilabel=False),
    "coco": DatasetInfo(image_size=124, n_classes=80, multilabel=True)
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


class CocoMulticlassDataset(Dataset):
    """
    Creates a COCOMultiClassDataset with labels on the form [category1, category2...]
    (not one hot)
    """

    def __init__(self, path, transform=None, target_transform=None, test=False):

        # initialize COCO api for instance annotations
        self.dataDir = f'{path}/coco'

        if test:
            self.dataType = 'val2014'
        else:
            self.dataType = 'train2014'

        annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)
        self.img_dir = f"{self.dataDir}/images/{self.dataType}"

        # initialize COCO api for instance annotations
        coco = COCO(annFile)

        cats = [cat['id'] for cat in coco.loadCats(coco.getCatIds())]
        cat_to_idx = dict(zip(cats, range(len(cats))))

        img_ids = coco.getImgIds()
        self.imgs = coco.loadImgs(img_ids)

        # lista av ans för varje bild
        anns_ids = [coco.getAnnIds(imgIds=imgid) for imgid in img_ids]

        self.labels = np.zeros((len(self.imgs), info['coco'].n_classes), dtype=np.int8)

        # Make one-hot encoded labels
        for i, ids in enumerate(anns_ids):
            for category_idx in map(lambda x: cat_to_idx.get(x['category_id']),
                                    coco.loadAnns(ids)):
                self.labels[i, category_idx] = 1

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx]['file_name'])
        image = Image.open(img_path)
        image = image.convert('RGB')

        label = self.labels[idx, :]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def coco(train, device, checkpoint):
    """Returns (dataloader, image_size)"""
    data_path = checkpoint["data_path"]
    limit = checkpoint["config"]["data"]["limit"]
    val_ratio = checkpoint["config"]["data"]["val_ratio"]
    batch_size = checkpoint["config"]["batch_size"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(info['coco'].image_size),
        transforms.RandomCrop((info['coco'].image_size, info['coco'].image_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if train:
        dataset = CocoMulticlassDataset(
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
        dataset = CocoMulticlassDataset(
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


def get_dataloader(dataset, train, device, checkpoint):
    if dataset == "cifar10":
        return cifar(train, device, checkpoint)
    elif dataset == "imagenette":
        return imagenette(train, device, checkpoint)
    elif dataset == "coco":
        return coco(train, device, checkpoint)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
