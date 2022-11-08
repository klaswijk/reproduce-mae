from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/data/datasets/coco.py
class CocoMultilabel(CocoDetection):
    NotImplemented


def cifar(train=True, batch_size=256):
    """Returns (dataloader, image_size)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    image_size = 32
    return dataloader, image_size