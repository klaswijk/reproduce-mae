import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils
from tqdm import tqdm

from data import cifar
from mae import small_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_PATH = "weights/cifar_mae.pth"

# Model
PATCH_SIZE = 4
MASK_RATIO = 0.5

# Optimizer
BATCH_SIZE = 16
LR = 0.001
BETA_1 = 0.9
BETA_2 = 0.999


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_comparison(input, output, mask, size=4):
    masked = torch.clone(input[:size])
    masked[:size, :, mask] = 0
    output[:size, :, ~mask] = input[:size, :, ~mask]  # Transfer known patches
    combined = torch.vstack([masked, input[:size], output[:size]])
    imshow(utils.make_grid(combined, nrow=size))


def mask_from_patches(masked_indices, image_size, patch_size):
    seq_length = image_size // patch_size
    mask = torch.ones(seq_length**2)
    mask[masked_indices] = 0
    mask = mask.reshape(seq_length, seq_length)
    return torch.nn.UpsamplingNearest2d(image_size)(mask[None, None])[0, 0].type(torch.bool)


def train(model_path=None, epochs=10, plot_example_interval=2):
    trainloader, image_size = cifar(train=True, batch_size=BATCH_SIZE)

    model = small_model(image_size, PATCH_SIZE, MASK_RATIO, model_path).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(BETA_1, BETA_2))

    losses = []
    for epoch in range(1, epochs):
        with tqdm(trainloader, unit="batch") as tepoch:
            for data, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data.to(DEVICE)
                optimizer.zero_grad()
                outputs, masked_indices = model(inputs)
                mask = mask_from_patches(masked_indices, image_size, PATCH_SIZE)
                loss = criterion(outputs[:, :, mask], inputs[:, :, mask])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                tepoch.set_postfix(ma_loss=f"{sum(losses[-10:]) / 10:.5f}")

        if epoch % plot_example_interval == 0:
            plot_comparison(inputs, outputs, mask)
            #val_loss = get_loss_from_dataloader(model, valloader, image_size, True)
            #print(f"Validation loss: {val_loss:.7f}")

    print(f"Saving model at '{model_path}'... ", end='')
    torch.save(model.state_dict(), model_path)
    print("Save sucessful!")

    plt.plot(losses)
    plt.savefig("loss.png")


def get_loss_from_dataloader(model, dataloader, image_size, plot=False):

    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        total_loss = 0
        for i, (data, _) in enumerate(dataloader):
            inputs = data.to(DEVICE)
            outputs, masked_indices = model(inputs)
            mask = mask_from_patches(masked_indices, image_size, PATCH_SIZE)
            loss = criterion(outputs[:, :, mask], inputs[:, :, mask])
            total_loss += loss.item()

        if plot:
            plot_comparison(inputs, outputs, mask)

    return total_loss / len(dataloader)


def test(model_path):
    testloader, image_size = cifar(train=False, batch_size=BATCH_SIZE)
    model = small_model(image_size, PATCH_SIZE, MASK_RATIO, model_path).to(DEVICE)
    print(f"Test loss: {get_loss_from_dataloader(model, testloader, image_size, True):.7f}")

if __name__ == "__main__":
    #train(model_path=WEIGHT_PATH, epochs=1000, plot_example_interval=100)
    test(model_path=WEIGHT_PATH)