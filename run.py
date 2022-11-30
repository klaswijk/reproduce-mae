import os
import datetime
import torch
import wandb

from torch.nn import MSELoss, CrossEntropyLoss, UpsamplingNearest2d
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, DataLoader

from mae import MAE
from data import info, get_dataloader
from plot import plot_reconstruction

import numpy as np


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


def log_reconstruction(epoch, input, output, mask, train):
    dataset_log = "train" if train else "val"
    dataset = "training" if train else "validation"
    if epoch == 1:
        images = wandb.Image(
            input[:4, :, :], caption=f"True images ({dataset} data)")
        wandb.log({f"{dataset_log}_reconstruction": images},
                  step=epoch)
    else:
        output[:4, :, ~mask] = input[:4, :, ~mask]
        images = wandb.Image(
            output[:4, :, :], caption=f"Reconstruction ({dataset} data)")
        wandb.log({f"{dataset_log}_reconstruction": images},
                  step=epoch)


def pretrain(checkpoint, epochs, device, checkpoint_frequency, id, log_image_ingerval):
    config = checkpoint["config"]
    patch_size = config["model"]["patch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]
    name = f"{id}_pretrain_{datetime.datetime.now()}"

    os.makedirs(
        f"{checkpoint['output_path']}/checkpoints/{name}", exist_ok=True)

    wandb.init(config=config, name=name, entity="mae_dd2412")

    trainloader, valloader = get_dataloader(dataset, True, device, checkpoint)
    train_reconstruction_loader = DataLoader(
        Subset(trainloader.dataset, range(4)), batch_size=4)
    criterion = MSELoss()

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    optimizer = Adam(model.parameters(), **config["optimizer"])
    scheduler = CosineAnnealingLR(optimizer, **config["scheduler"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    wandb.config.update({"number of train samples": len(
        trainloader.dataset), "number of validation samples": len(valloader.dataset)})

    start = checkpoint["pretrain_epoch"] + 1

    best_val_loss = (0, np.Inf)  # epoch and loss
    for epoch in range(start, start + epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0

        for input, _ in trainloader:
            input = input.to(device)
            optimizer.zero_grad()

            output, masked_indices = model(input)
            mask = mask_from_patches(
                masked_indices, image_size, patch_size)
            loss = criterion(input[:, :, mask], output[:, :, mask])

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        if epoch == 1 or epoch % log_image_ingerval == 0:
            with torch.no_grad():
                for input, _ in train_reconstruction_loader:
                    input = input.to(device)
                    output, masked_indices = model(input)
                    mask = mask_from_patches(
                        masked_indices, image_size, patch_size)
                    loss = criterion(input[:, :, mask], output[:, :, mask])
                    epoch_val_loss += loss.item()
            log_reconstruction(epoch, input, output, mask, True)

        with torch.no_grad():
            for input, _ in valloader:
                input = input.to(device)
                output, masked_indices = model(input)
                mask = mask_from_patches(
                    masked_indices, image_size, patch_size)
                loss = criterion(input[:, :, mask], output[:, :, mask])
                epoch_val_loss += loss.item()

        if epoch == 1 or epoch % log_image_ingerval == 0:
            log_reconstruction(epoch, input, output, mask, False)

        epoch_train_loss /= len(trainloader)
        epoch_val_loss /= len(valloader)
        wandb.log({"epoch": epoch,
                   "train_mse": epoch_train_loss,
                   "val_mse": epoch_val_loss,
                   "learning_rate": optimizer.param_groups[0]["lr"]
                   }, step=epoch)

        scheduler.step()

        if epoch % checkpoint_frequency == 0:
            save_checkpoint(
                f"{checkpoint['output_path']}/checkpoints/{name}/epoch_{epoch}.pth",
                checkpoint,
                random_state=torch.get_rng_state(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                pretrain_epoch=epoch,
            )

        if epoch > config["lookahead"] and epoch_val_loss < best_val_loss[1]:
            # found a better checkpoint
            save_checkpoint(
                f"{checkpoint['output_path']}/checkpoints/{name}/current_best.pth",
                checkpoint,
                random_state=torch.get_rng_state(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                pretrain_epoch=epoch,
            )
            best_val_loss = (epoch, epoch_val_loss)

        elif epoch-best_val_loss[0] > config["lookahead"]:
            print(
                f"Early stopping at \nEpoch={epoch} loss={epoch_val_loss} in favor of \nEpoch={best_val_loss[0]} loss={best_val_loss[1]}")
            # stopping based on how far we looked ahead
            # could load "current best" and save it with "earlystopping_{epoch}"
            return


def test_reconstruction(checkpoint, device):
    config = checkpoint["config"]
    batch_size = config["batch_size"]
    patch_size = config["model"]["patch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]

    os.makedirs(
        f"{checkpoint['output_path']}/plots/{dataset}/reconstruction/test/", exist_ok=True)

    testloader = get_dataloader(dataset, False, device, checkpoint)
    criterion = MSELoss()

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        total_loss = 0

        for input, _ in testloader:
            input = input.to(device)
            output, masked_indices = model(input)
            mask = mask_from_patches(masked_indices, image_size, patch_size)
            loss = criterion(input[:, :, mask], output[:, :, mask])
            total_loss += loss.item()

    epoch = checkpoint["pretrain_epoch"]
    plot_reconstruction(
        f"{checkpoint['output_path']}plots/{dataset}/reconstruction/test/epoch_{epoch}.png",
        input, output, mask
    )

    print(f"Test loss: {total_loss / len(testloader)}")


def finetune(checkpoint, epochs, device, checkpoint_frequency, id):
    config = checkpoint["config"]
    batch_size = config["batch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]
    name = id + "_finetune_" + str(datetime.datetime.now())

    os.makedirs(
        f"{checkpoint['output_path']}/checkpoints/{name}", exist_ok=True)

    wandb.init(config=config, name=name)

    trainloader, valloader = get_dataloader(dataset, True, device, checkpoint)
    criterion = CrossEntropyLoss()

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    optimizer = Adam(model.parameters(), **config["optimizer"])
    scheduler = CosineAnnealingLR(optimizer, **config["scheduler"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint["finetune_epoch"] != 0:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start = checkpoint["finetune_epoch"] + 1
    best_val_loss = (0, np.Inf)  # epoch and loss
    for epoch in range(start, start + epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0

        for input, target in trainloader:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model.classify(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            epoch_train_loss = loss.item()

        with torch.no_grad():
            for input, target in valloader:
                input = input.to(device)
                target = target.to(device)
                output = model.classify(input)
                loss = criterion(output, target)
                epoch_val_loss = loss.item()

        epoch_train_loss /= len(trainloader.dataset)
        epoch_val_loss /= len(valloader.dataset)
        wandb.log({"epoch": epoch, "train_ce": epoch_train_loss,
                  "val_ce": epoch_val_loss})

        scheduler.step()

        if epoch % checkpoint_frequency == 0:
            save_checkpoint(
                f"{checkpoint['output_path']}/checkpoints/{name}/epoch_{epoch}.pth",
                checkpoint,
                random_state=torch.get_rng_state(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                finetune_epoch=epoch,
            )

        if epoch > config["lookahead"] and epoch_val_loss < best_val_loss[1]:
            # found a better checkpoint
            save_checkpoint(
                f"{checkpoint['output_path']}/checkpoints/{name}/current_best.pth",
                checkpoint,
                random_state=torch.get_rng_state(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                finetune_epoch=epoch,
            )
            best_val_loss = (epoch, epoch_val_loss)

        elif epoch-best_val_loss[0] > config["lookahead"]:
            print(
                f"Early stopping at \nEpoch={epoch} loss={epoch_val_loss} in favor of \nEpoch={best_val_loss[0]} loss={best_val_loss[1]}")
            # stopping based on how far we looked ahead
            # could load "current best" and save it with "earlystopping_{epoch}"
            return


def test_classification(checkpoint, device):
    config = checkpoint["config"]
    batch_size = config["batch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]

    testloader = get_dataloader(dataset, False, device, checkpoint)

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    with torch.no_grad():
        for input, targets in testloader:
            input = input.to(device)
            target = target.to(device)
            output = model.classify(input)
            correct += torch.sum(output.argmax(dim=1) == targets)

    print(f"Test accuracy: {correct / len(testloader.dataset):.3f}")
