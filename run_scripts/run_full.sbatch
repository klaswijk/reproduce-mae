#!/usr/bin/env bash
#SBATCH --mem  16GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "smaug"
#SBATCH --time 60:00:00
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user rickym@kth.se
#SBATCH --output /Midgard/home/%u/logs/%J.out
#SBATCH --error  /Midgard/home/%u/logs/%J.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

. ~/miniconda3/etc/profile.d/conda.sh
conda activate mae 

echo "pre-traning"
python /Midgard/home/rickym/reproduce-mae/main.py --pretrain \
    --config /Midgard/home/rickym/reproduce-mae/configs/${config}_pretrain.yaml \
    --id ${SLURM_JOB_ID} \
    --epochs ${epochs_pretrain} \
    --checkpoint-frequency ${checkpoint_frequency} \
    --log_image_interval ${log_image_interval} \
    --data-path ${data_path} \
    --output-path ${output_path} \

echo "finetuning"
python /Midgard/home/rickym/reproduce-mae/main.py --finetune \
    --config configs/${config}_finetune.yaml \
    --checkpoint ${output_path}/checkpoints/${SLURM_JOB_ID}_pretrain/current_best.pth \
    --epochs ${epochs_finetune} \
    --id ${SLURM_JOB_ID} \
    --checkpoint-frequency ${checkpoint_frequency} \

echo "test classification"
python /Midgard/home/rickym/reproduce-mae/main.py --test-classification \
    --checkpoint ${output_path}/checkpoints/${SLURM_JOB_ID}_finetune/current_best.pth \
    --id ${SLURM_JOB_ID} \

echo "Finished"
