import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision import utils


def save_image(path, image):
    image = image / 2 + 0.5
    npimg = image.cpu().detach().numpy()
    plt.figure(dpi=200)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path)


def plot_reconstruction(
    path,
    true,
    reconstruction,
    mask,
    size=4,
):
    masked = torch.clone(true[:size])
    masked[:size, :, mask] = 0
    reconstruction[:size, :, ~mask] = true[:size,
                                           :, ~mask]  # Transfer known patches
    combined = torch.vstack([masked, true[:size], reconstruction[:size]])
    save_image(path, utils.make_grid(combined, nrow=size))


def plot_loss(path, loss):
    plt.figure(dpi=200)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.savefig(path)
