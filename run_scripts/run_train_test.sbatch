#!/usr/bin/env bash
#SBATCH --mem  16GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "smaug"
#SBATCH --time 60:00:00
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user rickym@kth.se
#SBATCH --output /Midgard/home/%u/logs/MAE_%J.out
#SBATCH --error  /Midgard/home/%u/logs/MAE_%J.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

/Midgard/home/rickym/miniconda3/bin/conda activate mae 

echo "pre-traning"
/Midgard/home/rickym/reproduce-mae/main.py --pretrain \
    --config /Midgard/home/rickym/reproduce-mae/configs/cifar10.yaml \
    --id ${SLURM_JOB_ID} \
    --epochs ${epochs} \
    --checkpoint-frequency ${checkpoint-frequency} \
    --log_image_ingerval ${log_image_ingerval} \ 
    --data-path ${data-path} \ 
    --output-path ${output-path} \ 


echo "Finished"
