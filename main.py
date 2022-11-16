import os
import argparse
import yaml
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import run

from mae import MAE
from data import info


def initialize(config):
    torch.manual_seed(config["random_seed"])
    image_size, n_classes = info[config["data"]["dataset"]]
    model = MAE(image_size, n_classes, **config["model"])
    optimizer = Adam(model.parameters(), **config["optimizer"])
    scheduler = ExponentialLR(optimizer, **config["scheduler"])
    return {
        "config": config,
        "random_state": torch.get_rng_state(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "pretrain_epoch": 0,
        "pretrain_training_loss": [],
        "finetune_epoch": 0,
        "finetune_training_loss": [],
    }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint-frequency", type=int, default=100)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrain", action="store_true")
    group.add_argument("--test-reconstruction", action="store_true")
    group.add_argument("--finetune", action="store_true")
    group.add_argument("--test-classification", action="store_true")
    return parser.parse_args()


def main():
    for dir in ["configs", "checkpoints", "plots"]:
        os.makedirs(dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()

    if args.checkpoint:
        # Continue from checkpoint if specified
        checkpoint = torch.load(args.checkpoint)
        config = checkpoint["config"]
        torch.set_rng_state(checkpoint["random_state"])
    elif args.config:
        # Start fresh based on config
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        checkpoint = initialize(config)

    # Run the specified task
    if args.pretrain:
        run.pretrain(
            checkpoint, 
            args.epochs, 
            device, 
            args.checkpoint_frequency
        )
    elif args.test_reconstruction:
        run.test_reconstruction(
            checkpoint, 
            device,
        )
    elif args.finetune:
        run.finetune(
            checkpoint, 
            args.epochs, 
            device, 
            args.checkpoint_frequency
        )
    elif args.test_classification:
        run.test_classification(
            checkpoint, 
            device,
        )


if __name__ == "__main__":
    main()