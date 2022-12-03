#!/bin/bash

echo "Starting job"

echo "pre-traning $1"
python ./main.py --pretrain \
    --config ./configs/$2_pretrain.yaml \
    --id $1 \
    --epochs $3 \
    --checkpoint-frequency $5 \
    --log_image_ingerval $6 \
    --data-path ./data \
    --output-path ./ \

echo "finetuning"
python ./main.py --finetune \
    --config ./configs/$2_finetune.yaml \
    --checkpoint ./checkpoints/${1}_pretrain/current_best.pth \
    --epochs $4 \
    --id $1 \
    --checkpoint-frequency $5 \

echo "test classification"
python ./main.py --test-classification \
    --checkpoint ./checkpoints/$1_finetune/current_best.pth \
    --id $1 \

echo "Finished"


# example 
# id, config, pre epoch, fine epoch, checkpoint freq, log image freq 
# bash ./run_scripts/run_full_local.sh dev imagenette_testing 100 100 10 2
# bash ./run_scripts/run_full_local.sh dev coco_testing 100 100 10 2
# bash ./run_scripts/run_full_local.sh dev imagenette 2000 2000 500 20
# bash ./run_scripts/run_full_local.sh dev coco 4000 4000 500 20
