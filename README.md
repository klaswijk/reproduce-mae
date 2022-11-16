# reproduce-mae
Reproduction of "Masked Autoencoders Are Scalable Vision Learners"

## Example Usage
Pretrain
```
python main.py --pretrain --config config/cifar10.yaml --epochs 100 --checkpoint-frequency 10
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

