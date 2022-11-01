import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from mae import MAE

MODEL_PATH = "./cifar_mae.pth"


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    model = MAE(
        image_size=32,
        patch_size=4,
        encoder_layers=12,
        encoder_num_heads=4,
        encoder_hidden_dim=12,
        encoder_mlp_dim=12,
        decoder_layers=4,
        decoder_num_heads=4,
        decoder_hidden_dim=12,
        decoder_mlp_dim=12,
        mask_ratio=0.5,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(10):

        running_loss = 0.0
        for i, (data, _) in enumerate(trainloader):
            inputs = data.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')

        imshow(utils.make_grid(torch.vstack([inputs[:4], outputs[:4]]), nrow=4))
        print(f"Finished epoch: {epoch}")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved model")


if __name__ == "__main__":
    main()
