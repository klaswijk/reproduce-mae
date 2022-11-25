import os
import datetime
import torch
import wandb

from torch.nn import MSELoss, CrossEntropyLoss, UpsamplingNearest2d
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from mae import MAE
from data import info, get_dataloader
from plot import plot_reconstruction


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


def pretrain(checkpoint, epochs, device, checkpoint_frequency, id, log_image_ingerval):
    config = checkpoint["config"]
    patch_size = config["model"]["patch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes = info[dataset]

    os.makedirs(
        f"{checkpoint['output_path']}/checkpoints/{dataset}/pretrain", exist_ok=True)

    wandb.init(config=config, name=id + "_pretrain_" +
               str(datetime.datetime.now()))

    trainloader, valloader = get_dataloader(dataset, True, device, checkpoint)
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

        with torch.no_grad():
            for input, _ in valloader:
                input = input.to(device)
                output, masked_indices = model(input)
                mask = mask_from_patches(
                    masked_indices, image_size, patch_size)
                loss = criterion(input[:, :, mask], output[:, :, mask])
                epoch_val_loss += loss.item()

        if epoch == 1:
            images = wandb.Image(input[:4, :, :], caption="True images")
            wandb.log({"reconstruction": images},
                      step=epoch)

        if epoch % log_image_ingerval == 0:
            output[:4, :, ~mask] = input[:4, :, ~mask]
            images = wandb.Image(output[:4, :, :], caption="Reconstruction")
            wandb.log({"reconstruction": images},
                      step=epoch)

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
                f"{checkpoint['output_path']}/checkpoints/{dataset}/pretrain/epoch_{epoch}.pth",
                checkpoint,
                random_state=torch.get_rng_state(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                pretrain_epoch=epoch,
            )


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

    os.makedirs(
        f"{checkpoint['output_path']}/checkpoints/{dataset}/finetune", exist_ok=True)

    wandb.init(config=config, name=id + "_finetune_" +
               str(datetime.datetime.now()))

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

        epoch_train_loss /= len(trainloader)
        epoch_val_loss /= len(valloader)
        wandb.log({"epoch": epoch, "train_ce": epoch_train_loss,
                  "val_ce": epoch_val_loss})

        scheduler.step()

        if epoch % checkpoint_frequency == 0:
            save_checkpoint(
                f"{checkpoint['output_path']}/checkpoints/{dataset}/finetune/epoch_{epoch}.pth",
                checkpoint,
                random_state=torch.get_rng_state(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                finetune_epoch=epoch,
            )


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
    total = 0
    with torch.no_grad():
        for input, targets in testloader:
            input = input.to(device)
            target = target.to(device)
            output = model.classify(input)
            correct += torch.sum(output.argmax(dim=1) == targets)
            total += len(targets)

    print(f"Test accuracy: {correct / total:.3f}")
