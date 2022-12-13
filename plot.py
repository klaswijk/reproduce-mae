import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from torchvision import utils


def save_image(path, image):
    image = image / 2 + 0.5
    npimg = image.cpu().detach().numpy()
    plt.figure(dpi=200)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')


def plot_reconstruction(
    path,
    true,
    reconstruction,
    mask,
    size=10,
):
    masked = torch.clone(true[:size])
    masked[:, :, mask] = 0
    reconstruction[:size, :, ~mask] = true[:size, :, ~mask]  # Transfer known patches

    for i in range(size):
        img_path = path + f'/img{i}'
        os.makedirs(img_path, exist_ok=True)

        save_image(img_path + "/true.png", true[i])

        recon = torch.stack([masked[i], reconstruction[i]])
        save_image(img_path + "/reconstruction.png",
                   utils.make_grid(recon, nrow=2, padding=0))

        combined = torch.stack([masked[i], reconstruction[i], true[i]])
        save_image(img_path + "/combined.png", utils.make_grid(combined, nrow=3, padding=0))


def plot_loss(path, loss):
    plt.figure(dpi=200)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.savefig(path)
