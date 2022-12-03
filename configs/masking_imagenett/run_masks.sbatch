#!/usr/bin/env bash
#SBATCH --mem  16GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --constrain "smaug"
#SBATCH --time 120:00:00
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user rickym@kth.se
#SBATCH --output /Midgard/home/%u/logs/%J.out
#SBATCH --error  /Midgard/home/%u/logs/%J.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

. ~/miniconda3/etc/profile.d/conda.sh
conda activate mae 

for config in mask00 mask03 mask05 mask07 mask09 
do
    echo "pre-traning $config"
    python /Midgard/home/rickym/reproduce-mae/main.py --pretrain \
        --config /Midgard/home/rickym/reproduce-mae/configs/masking_imagenett/${config}_pretrain.yaml \
        --id ${SLURM_JOB_ID}_${config} \
        --epochs ${epochs_pretrain} \
        --checkpoint-frequency ${checkpoint_frequency} \
        --log_image_ingerval ${log_image_ingerval} \
        --data-path ${data_path} \
        --output-path ${output_path} \

    echo "finetuning $config"
    python /Midgard/home/rickym/reproduce-mae/main.py --finetune \
        --config configs/masking_imagenett/${config}_finetune.yaml \
        --checkpoint ${output_path}/checkpoints/${SLURM_JOB_ID}_${config}_pretrain/current_best.pth \
        --epochs ${epochs_finetune} \
        --id ${SLURM_JOB_ID}_${config} \
        --checkpoint-frequency ${checkpoint_frequency} \

    echo "test classification $config"
    python /Midgard/home/rickym/reproduce-mae/main.py --test-classification \
        --checkpoint ${output_path}/checkpoints/${SLURM_JOB_ID}_${config}_finetune/current_best.pth \
        --id ${SLURM_JOB_ID}_${config} \

    echo "Finished $config"
done