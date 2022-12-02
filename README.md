# reproduce-mae
Reproduction of "Masked Autoencoders Are Scalable Vision Learners"

## Example Usage
Pretrain
```
python main.py --pretrain --config configs/cifar10.yaml --epochs 100 --checkpoint-frequency 10
python main.py --pretrain --config configs/imagenette.yaml --epochs 10 --id dev --checkpoint-frequency 10 --log_image_ingerval 2

python main.py --pretrain --config configs/imagenette_pretrain.yaml --epochs 2000 --id dev_large --checkpoint-frequency 10000 --log_image_ingerval 10

```

# test iamgenetter 
```
python main.py --pretrain --config configs/imagenette_testing_pretrain.yaml --epochs 100 --id dev --checkpoint-frequency 1 --log_image_ingerval 1

python main.py --finetune --config configs/imagenette_testing_finetune.yaml --checkpoint /home/morningstar/Documents/github/reproduce-mae/checkpoints/dev_pretrain/current_best.pth --epochs 100 --id dev --checkpoint-frequency 1

python main.py --test-classification --id dev --checkpoint /home/morningstar/Documents/github/reproduce-mae/checkpoints/dev_finetune/current_best.pth
```

# test coco
```
python main.py --pretrain --config configs/coco_testing_pretrain.yaml --epochs 100 --id dev --checkpoint-frequency 1 --log_image_ingerval 1

python main.py --finetune --config configs/coco_testing_finetune.yaml --checkpoint /home/morningstar/Documents/github/reproduce-mae/checkpoints/dev_pretrain/current_best.pth --epochs 100 --id dev --checkpoint-frequency 1

python main.py --test-classification --id dev --checkpoint /home/morningstar/Documents/github/reproduce-mae/checkpoints/dev_finetune/current_best.pth
```



Test reconstruction
```
python main.py --test-reconstruction --checkpoint checkpoints/cifar10_pretrain_epoch_100.pth
```
Finetune
```
python main.py --finetune --checkpoint checkpoints/cifar10_pretrain_epoch_100.pth --epochs 100 --checkpoint-frequency 10
```
Test classification
```
python main.py --test-classification --checkpoint checkpoints/cifar10_finetune_epoch_100.pth 
```
