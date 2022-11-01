import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from mae import MAE

MODEL_PATH = "./cifar_mae.pth"


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
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

    trainloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    model = MAE(
        image_size=32,
        patch_size=4,
        encoder_layers=12,
        encoder_num_heads=4,
        encoder_hidden_dim=12,
        encoder_mlp_dim=12,
        decoder_layers=12,
        decoder_num_heads=4,
        decoder_hidden_dim=12,
        decoder_mlp_dim=12,
        mask_ratio=0.5,
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, _ = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

                imshow(utils.make_grid(torch.vstack([inputs, outputs]), nrow=4))

    print("Finished training")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved model")


if __name__ == "__main__":
    main()
