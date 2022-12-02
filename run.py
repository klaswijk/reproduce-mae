import os
import datetime
import torch
import wandb

from torch.nn import MSELoss, CrossEntropyLoss, BCELoss, UpsamplingNearest2d
from torch.optim import AdamW
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
    image_size, n_classes, _ = info[dataset]
    name = f"{id}_pretrain"

    os.makedirs(
        f"{checkpoint['output_path']}/checkpoints/{name}", exist_ok=True)

    wandb.init(config=config, name=name + "_" +
               str(datetime.datetime.now()), entity="mae_dd2412")

    trainloader, valloader = get_dataloader(
        dataset, True, device, checkpoint, transform_type="pretrain")
    train_reconstruction_loader = DataLoader(
        Subset(trainloader.dataset, range(4)), batch_size=4)
    criterion = MSELoss()

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    optimizer = AdamW(model.parameters(), **config["optimizer"])
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

            input_patches = model.patch(input)
            output_patches, masked_indices = model(input)

            loss = criterion(
                input_patches[:, :, masked_indices], output_patches[:, :, masked_indices])

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        if epoch == 1 or epoch % log_image_ingerval == 0:
            with torch.no_grad():
                for input, _ in train_reconstruction_loader:
                    input = input.to(device)
                    output_patches, masked_indices = model(input)
                    output = model.unpatch(output_patches)
                    mask = mask_from_patches(
                        masked_indices, image_size, patch_size)
            log_reconstruction(epoch, input, output, mask, True)

        with torch.no_grad():
            for input, _ in valloader:
                input = input.to(device)

                input_patches = model.patch(input)
                output_patches, masked_indices = model(input)

                loss = criterion(
                    input_patches[:, :, masked_indices], output_patches[:, :, masked_indices])

                epoch_val_loss += loss.item()

        if epoch == 1 or epoch % log_image_ingerval == 0:
            mask = mask_from_patches(
                masked_indices, image_size, patch_size)
            output = model.unpatch(output_patches)
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
    image_size, n_classes, _ = info[dataset]

    os.makedirs(
        f"{checkpoint['output_path']}/plots/{dataset}/reconstruction/test/", exist_ok=True)

    testloader = get_dataloader(
        dataset, False, device, checkpoint, transform_type=None)
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
    image_size, n_classes, multilabel = info[dataset]
    name = id + "_finetune"

    os.makedirs(
        f"{checkpoint['output_path']}/checkpoints/{name}", exist_ok=True)

    wandb.init(config=config, name=name+"_"+str(datetime.datetime.now()))

    trainloader, valloader = get_dataloader(
        dataset, True, device, checkpoint, transform_type="finetune")
    if multilabel:
        activate = torch.nn.Sigmoid()
        criterion = BCELoss()
    else:
        activate = torch.nn.Identity()
        criterion = CrossEntropyLoss(label_smoothing=0.1)

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    optimizer = AdamW(model.parameters(), **config["optimizer"])
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

            loss = criterion(activate(output), target)

            loss.backward()
            optimizer.step()

            epoch_train_loss = loss.item()

        with torch.no_grad():
            for input, target in valloader:
                input = input.to(device)
                target = target.to(device)
                output = model.classify(input)
                loss = criterion(activate(output), target)
                epoch_val_loss = loss.item()

        epoch_train_loss /= len(trainloader.dataset)
        epoch_val_loss /= len(valloader.dataset)
        wandb.log({"epoch": epoch, "train_loss": epoch_train_loss,
                  "val_loss": epoch_val_loss})

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


def test_classification(checkpoint, device, id):
    config = checkpoint["config"]
    batch_size = config["batch_size"]
    dataset = config["data"]["dataset"]
    image_size, n_classes, multilabel = info[dataset]
    name = id + "_test_classification"

    wandb.init(config=config, name=name+"_"+str(datetime.datetime.now()))

    testloader = get_dataloader(
        dataset, False, device, checkpoint, transform_type=None)

    model = MAE(image_size, n_classes, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if multilabel:
        activate = torch.nn.Sigmoid()
        criterion = BCELoss()

        def number_correct(p, t, type_of_correct="tp"):

            t = t.type(torch.bool)

            correct = (p >= 0.5) == t
            not_correct = ~correct
            if type_of_correct == "tp":
                out = correct*t  # is correct then mask for true
            elif type_of_correct == "tn":
                out = correct*(~t)  # is correct then mask for false
            elif type_of_correct == "fp":
                out = not_correct*t  # is not correct then mask for true
            elif type_of_correct == "fn":
                out = not_correct*(~t)  # is not correct then mask for false
            return torch.sum(out)

    else:
        activate = torch.nn.Identity()
        criterion = CrossEntropyLoss()
        def number_correct(p, t): return torch.sum(p.argmax(dim=1) == t)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    amount_active = 0
    total_loss = 0
    with torch.no_grad():
        for input, target in testloader:
            input = input.to(device)
            target = target.to(device)
            output = model.classify(input)
            loss = criterion(activate(output), target)
            total_loss += loss.item()
            tp += number_correct(output, target)
            if multilabel:
                tn += number_correct(output, target, "tn")
                fp += number_correct(output, target, "fp")
                fn += number_correct(output, target, "fn")
            amount_active += torch.sum(target)

    if multilabel:
        test_loss = total_loss / len(testloader)
        res = {"test_loss": test_loss, "true positive": tp,
               "true negative": tn, "false positive": fp, "false negative": fn}

        wandb.log(res)
        print(res)

    else:
        test_loss = total_loss / len(testloader)
        test_acc = tp / (len(testloader.dataset))

        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        print(f"Test loss: {test_loss} Test accuracy: {test_acc}")
