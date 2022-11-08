import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from mae import MAE

MODEL_PATH = "./cifar_mae.pth"


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def mask_from_patches(masked_indices, image_size, patch_size):
    seq_length = image_size // patch_size
    mask = torch.zeros(seq_length**2)
    mask[masked_indices] = 1
    mask = mask.reshape(seq_length, seq_length)
    return torch.nn.UpsamplingNearest2d(image_size)(mask[None, None])[0, 0].type(torch.bool)


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

    trainloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    image_size = 32
    patch_size = 4

    model = MAE(
        image_size=image_size,
        patch_size=patch_size,
        encoder_layers=8,
        encoder_num_heads=8,
        encoder_hidden_dim=16,
        encoder_mlp_dim=64,
        decoder_layers=4,
        decoder_num_heads=4,
        decoder_hidden_dim=16,
        decoder_mlp_dim=64,
        mask_ratio=0.5,
    ).to(device)

    #model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    for epoch in range(25):

        running_loss = 0.0
        for i, (data, _) in enumerate(trainloader):
            inputs = data.to(device)

            optimizer.zero_grad()
            outputs, masked_indices = model(inputs)
            mask = mask_from_patches(masked_indices, image_size, patch_size)
            loss = criterion(outputs[:, :, ~mask], inputs[:, :, ~mask])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')

        outputs[:4, :, mask] = inputs[:4, :, mask]  # Transfer known patches
        imshow(utils.make_grid(torch.vstack([inputs[:4], outputs[:4]]), nrow=4))
        print(f"Finished epoch: {epoch}")
        #torch.save(model.state_dict(), MODEL_PATH)
        print("Saved model")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    image_size = 32
    patch_size = 4

    model = MAE(
        image_size=image_size,
        patch_size=patch_size,
        encoder_layers=8,
        encoder_num_heads=8,
        encoder_hidden_dim=16,
        encoder_mlp_dim=64,
        decoder_layers=4,
        decoder_num_heads=4,
        decoder_hidden_dim=8,
        decoder_mlp_dim=32,
        mask_ratio=0.5,
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    testloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    data, _ = next(iter(testloader))
    testinputs = data.to(device)
    testoutputs, m = model(testinputs.to(device))
    mask = mask_from_patches(m, image_size, patch_size)
    testoutputs[:4, :, mask] = testinputs[:4, :, mask]  # Transfer known patches
    maskinputs = torch.clone(testinputs)
    maskinputs[:4, :, mask] = 0
    imshow(utils.make_grid(torch.vstack([maskinputs, testinputs, testoutputs]), nrow=4))


if __name__ == "__main__":
    main()
