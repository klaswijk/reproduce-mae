import os
import torch

from torch.nn import MSELoss, CrossEntropyLoss, UpsamplingNearest2d
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from mae import MAE
from data import info, get_dataloader
from plot import plot_reconstruction, plot_loss


def mask_from_patches(masked_indices, image_size, patch_size):
    seq_length = image_size // patch_size
    mask = torch.ones(seq_length**2)
    mask[masked_indices] = 0
    mask = mask.reshape(seq_length, seq_length)
    return UpsamplingNearest2d(image_size)(mask[None, None])[0, 0].type(torch.bool)


def save_checkpoint(path, checkpoint, **updates):
    for key, value in updates.items():
        checkpoint[key] = value
    torch.save(checkpoint, path)


def pretrain(checkpoint, epochs, device, checkpoint_frequency):
    config = checkpoint["config"]
    batch_size = config["batch_size"]
    patch_size = config["model"]["patch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]

    os.makedirs(f"checkpoints/{dataset}/pretrain", exist_ok=True)
    os.makedirs(f"plots/{dataset}/reconstruction/train/", exist_ok=True)
    os.makedirs(f"plots/{dataset}/loss/pretrain", exist_ok=True)

    trainloader = get_dataloader(dataset, True, batch_size, device, config["data"]["limit"])
    criterion = MSELoss()

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    optimizer = Adam(model.parameters(), **config["optimizer"])
    scheduler = ExponentialLR(optimizer, **config["scheduler"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])    

    training_loss = checkpoint["pretrain_training_loss"]

    start = checkpoint["pretrain_epoch"] + 1
    for epoch in range(start, start + epochs):
        epoch_loss = 0
        with tqdm(trainloader, unit="batches") as pbar:
            for i, (input, _) in enumerate(pbar, start=1):
                optimizer.zero_grad()

                output, masked_indices = model(input)
                mask = mask_from_patches(masked_indices, image_size, patch_size)
                loss = criterion(input[:, :, mask], output[:, :, mask])

                loss.backward()
                optimizer.step()

                training_loss.append(loss.item())
                epoch_loss = (epoch_loss * (i-1) + loss.item()) / i

                pbar.set_description(f"Epoch {epoch:4d}")
                pbar.set_postfix(MSE=f"{epoch_loss:.5f}")

        scheduler.step()

        if epoch % checkpoint_frequency == 0:
            
            save_checkpoint(
                f"checkpoints/{dataset}/pretrain/epoch_{epoch}.pth",
                checkpoint,
                random_state=torch.get_rng_state(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                pretrain_epoch=epoch,
                pretrain_training_loss=training_loss
            )
            plot_reconstruction(
                f"plots/{dataset}/reconstruction/train/epoch_{epoch}.png",
                input, output, mask
            )
    
    plot_loss(
        f"plots/{dataset}/loss/pretrain/epoch_{epoch}.png",
        training_loss
    )


def test_reconstruction(checkpoint, device):
    config = checkpoint["config"]
    batch_size = config["batch_size"]
    patch_size = config["model"]["patch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]

    os.makedirs(f"plots/{dataset}/reconstruction/test/", exist_ok=True)

    testloader = get_dataloader(dataset, False, batch_size, device)
    criterion = MSELoss()

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        total_loss = 0

        for input, _ in testloader:
            output, masked_indices = model(input)
            mask = mask_from_patches(masked_indices, image_size, patch_size)
            loss = criterion(input[:, :, mask], output[:, :, mask])
            total_loss += loss.item()
    
    epoch = checkpoint["pretrain_epoch"]
    plot_reconstruction(
            input, output, mask, 
            path=f"plots/{dataset}/reconstruction/test/epoch_{epoch}.png"
    )

    print(f"Test loss: {total_loss / len(testloader)}")


def finetune(checkpoint, epochs, device, checkpoint_frequency):
    config = checkpoint["config"]
    batch_size = config["batch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]

    os.makedirs(f"checkpoints/{dataset}/finetune", exist_ok=True)
    os.makedirs(f"plots/{dataset}/loss/finetune/", exist_ok=True)

    trainloader = get_dataloader(
        dataset, True, batch_size, device, config["data"]["limit"]
    )
    criterion = CrossEntropyLoss()

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    optimizer = Adam(model.parameters(), **config["optimizer"])
    scheduler = ExponentialLR(optimizer, **config["scheduler"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])    
    if checkpoint["finetune_epoch"] != 0:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    training_loss = checkpoint["finetune_training_loss"]

    start = checkpoint["finetune_epoch"] + 1
    for epoch in range(start, start + epochs):
        epoch_loss = 0
        with tqdm(trainloader, unit="batches") as pbar:
            for i, (input, target) in enumerate(pbar, start=1):
                optimizer.zero_grad()

                output = model.classify(input)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                training_loss.append(loss.item())
                epoch_loss = (epoch_loss * (i-1) + loss.item()) / i

                pbar.set_description(f"Epoch {epoch:4d}")
                pbar.set_postfix(CE=epoch_loss)

        scheduler.step()

        if epoch % checkpoint_frequency == 0:
            save_checkpoint(
                f"checkpoints/{dataset}/finetune/epoch_{epoch}.pth",
                checkpoint,
                random_state=torch.get_rng_state(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                finetune_epoch=epoch,
                finetune_training_loss=training_loss
            )
    
    plot_loss(
        f"plots/{dataset}/loss/finetune/epoch_{epoch}.png",
        training_loss
    )


def test_classification(checkpoint, device):
    config = checkpoint["config"]
    batch_size = config["batch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]

    testloader = get_dataloader(dataset, False, batch_size, device)

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for input, targets in testloader:
            output = model.classify(input)
            correct += torch.sum(output.argmax(dim=1) == targets)
            total += len(targets)

    print(f"Test accuracy: {correct / total:.3f}")