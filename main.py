import argparse
import os

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import run
from data import info
from mae import MAE


def initialize(config):
    torch.manual_seed(config["random_seed"])
    image_size, n_classes, _ = info[config["data"]["dataset"]]
    model = MAE(image_size, n_classes, **config["model"])
    optimizer = AdamW(model.parameters(), **config["optimizer"])
    scheduler = CosineAnnealingLR(optimizer, **config["scheduler"])
    return {
        "config": config,
        "random_state": torch.get_rng_state(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "pretrain_epoch": 0,
        "finetune_epoch": 0,
    }


def overwrite(checkpoint, config):
    """Overwrite the optimizer and sheduler"""
    image_size, n_classes, _ = info[config["data"]["dataset"]]
    model = MAE(image_size, n_classes, **config["model"])
    optimizer = AdamW(model.parameters(), **config["optimizer"])
    scheduler = CosineAnnealingLR(optimizer, **config["scheduler"])
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    return checkpoint


def parse_arguments():
    parser = argparse.ArgumentParser()
    settings = parser.add_argument_group()
    settings.add_argument("--config")
    settings.add_argument("--checkpoint")
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--pretrain", action="store_true")
    task.add_argument("--test-reconstruction", action="store_true")
    task.add_argument("--finetune", action="store_true")
    task.add_argument("--test-classification", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint-frequency", type=int, default=100)
    parser.add_argument("--id")
    parser.add_argument("--log_image_interval", type=int, default=100)
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--output-path", default="./")
    parser.add_argument("--data-in-memory", default=False, action="store_true")
    parser.add_argument("--override-paths", default=False, action="store_true")
    parser.add_argument("--finetune-blocks", type=int, default=None)
    return parser.parse_args()


def main():
    print("Starting main!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    args = parse_arguments()

    for dir in ["checkpoints", "plots"]:
        os.makedirs(f"{args.output_path}{dir}", exist_ok=True)

    if args.checkpoint:
        # Continue from checkpoint if specified
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint["config"]
        torch.set_rng_state(checkpoint["random_state"].to("cpu"))

    if args.config:
        # Start fresh based on config
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        if not args.checkpoint:
            checkpoint = initialize(config)
            checkpoint["data_path"] = args.data_path
            checkpoint["output_path"] = args.output_path
        else:
            checkpoint["config"] = config
            checkpoint = overwrite(checkpoint, config)

    if args.override_paths:
        print("overriding paths")
        checkpoint["data_path"] = args.data_path
        checkpoint["output_path"] = args.output_path

    # Run the specified task
    if args.pretrain:
        run.pretrain(
            checkpoint,
            args.epochs,
            device,
            args.checkpoint_frequency,
            args.id,
            args.log_image_interval,
            args.data_in_memory
        )
    elif args.test_reconstruction:
        run.test_reconstruction(
            checkpoint,
            device,
            args.data_in_memory,
        )
    elif args.finetune:
        run.finetune(
            checkpoint,
            args.epochs,
            device,
            args.checkpoint_frequency,
            args.id,
            args.data_in_memory,
            args.finetune_blocks
        )
    elif args.test_classification:
        run.test_classification(
            checkpoint,
            device,
            args.id,
            args.data_in_memory
        )


if __name__ == "__main__":
    main()
