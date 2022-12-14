# reproduce-mae

Reproduction of "Masked Autoencoders Are Scalable Vision Learners" 🤿😷🪱

https://arxiv.org/pdf/2111.06377.pdf

The software is split into 4 sections

# Table of contents

1. [Datasets 💾](#datasets)
    1. [Imagenette 🚚](#nette)
    2. [Imagewoof 🐕](#woof)
    3. [COCO 🐬](#coco)
    4. [Cifar10 🙈](#cifar)
2. [Weights and biases ⚖️](#wandb)
3. [Traning 🏋️](#traning)
    1. [Pre-training 👩‍🍼](#pretraining)
    2. [Fine-tuning 🧑‍🎓](#finetuning)
4. [Testing 🧑‍🏫](#testing)
    1. [Test-classification 📖](#testclassification)
    2. [Test-reconstruction 📚](#testreconstruction)

# Datasets 💾 <a name="datasets"></a>

All datasets need to be downloaded and put into a separate folder, default is "./data" but can be override with "
--data-path <new-path>" argument

### Imagenette 🚚 <a name="nette"></a>

Can be downloaded at https://github.com/fastai/imagenette#imagenette-1
Current implementation is tested on 160px size

### Imagewoof 🐕 <a name="woof"></a>

Can be downloaded at https://github.com/fastai/imagenette#imagewoof
Current implementation is tested on 160px size

### Coco 🐬 <a name="coco"></a>

Can be downloaded at with ```bash ./download_coco.sh <version>``` with version 2014 or 2017
Current implementation is tested on coco2017 and is then resized using

```
python resize_coco.py --size 120 --full-size-path ./data/coco/ --resized-path ./data/coco-small/
```

### Cifar10 🙈 <a name="cifar"></a>

Will be downloaded automatically if not downloaded in advance.

# Weights and biases ⚖️ <a name="wandb"></a>

Weights and biases is used to log data during all the runs, instructions on how to install the software can be found at
on https://wandb.ai/

# Training 🏋️

### Pre-training👩‍🍼 <a name="pretraining"></a>

We have used .yaml files to define the structure of the model and then specifics for the runs with arguments to the
main.py file.

Example runs:

```
python main.py --pretrain --config configs/cifar10.yaml --epochs 100 --id dev --checkpoint-frequency 10 --log_image_interval 2
python main.py --pretrain --config configs/imagenette_pretrain.yaml --epochs 100 --id dev --checkpoint-frequency 10 --log_image_interval 2
```

## fine tuning 🧑‍🎓 <a name="finetuning"></a>

Fine tuning is used to train for classification. The software will use the config from the checkpoint if no new is
given,
otherwise a new instance of the model is created if a config is given.

Example of ViT model

```
python main.py --finetune --config configs/imagenette_finetune.yaml --epochs 100 --id dev --checkpoint-frequency 10
```

Example of MAE model that is pretrained but we change learning rate scheduler

```
python main.py --finetune --config configs/imagenette_finetune.yaml --checkpoint ./checkpoints/<run_id>_pretrain/current_best.pth --epochs 100 --id dev --checkpoint-frequency 10
```

Example of MAE model with same configs as pretrain

```
python main.py --finetune --checkpoint ./checkpoints/<old_run_id>_pretrain/current_best.pth --epochs 100 --id dev --checkpoint-frequency 10
```

## Testing 🧑‍🏫

Given a checkpoint we can conduct two tests on the data, this is done on the part of the data that is not used during
training.

### test-classification 📖 <a name="testclassification"></a>

Example of test classification

```
python main.py --test-classification --id dev --checkpoint ./checkpoints/<run_id>_finetune/current_best.pth
```

## test-reconstruction 📚 <a name="testreconstruction"></a>

Example of test reconstruction

```
python main.py --test-reconstruction --checkpoint checkpoints/<run_id>_pretrain/current_best.pth
```
