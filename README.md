# reproduce-mae
Reproduction of "Masked Autoencoders Are Scalable Vision Learners"

## Example Usage
Pretrain
```
python main.py --pretrain --config configs/cifar10.yaml --epochs 100 --checkpoint-frequency 10
python main.py --pretrain --config configs/imagenette.yaml --epochs 10 --id dev --checkpoint-frequency 10 --log_image_ingerval 2
```

# test 
```
python main.py --pretrain --config configs/imagenette_test.yaml --epochs 100 --id dev_pretrain --checkpoint-frequency 1 --log_image_ingerval 1

python main.py --finetune --config configs/imagenette_test_finetune.yaml --checkpoint '/home/morningstar/Documents/github/reproduce-mae/checkpoints/dev_pretrain_2022-11-30 15:10:19.203581/current_best.pth' --epochs 100 --id dev_finetune --checkpoint-frequency 1

python main.py --finetune --checkpoint '/home/morningstar/Documents/github/reproduce-mae/checkpoints/dev_pretrain_2022-11-30 15:10:19.203581/current_best.pth' --epochs 100 --id dev_finetune --checkpoint-frequency 1
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
