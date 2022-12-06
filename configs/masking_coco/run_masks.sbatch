#!/usr/bin/env bash
#SBATCH --mem  16GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "smaug"
#SBATCH --time 120:00:00
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user rickym@kth.se
#SBATCH --output /Midgard/home/%u/logs/%J.out
#SBATCH --error  /Midgard/home/%u/logs/%J.err


#call with: sbatch --export epochs_pretrain=4000,epochs_finetune=4000,checkpoint_frequency=500,log_image_ingerval=50,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models /Midgard/home/rickym/reproduce-mae/configs/masking_imagenette/run_masks.sbatch
#call with: sbatch /Midgard/home/rickym/reproduce-mae/configs/masking_imagenette/run_masks.sbatch
# 346585

epochs_pretrain=4000
epochs_finetune=4000
checkpoint_frequency=500
log_image_ingerval=50
data_path=/local_storage/datasets/rickym/
output_path=/local_storage/users/rickym/models

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "epochs_pretrain: $epochs_pretrain"
echo "epochs_finetune: $epochs_finetune"
echo "checkpoint_frequency: $checkpoint_frequency"
echo "log_image_interval: $log_image_ingerval"
echo "data_path: $data_path"
echo "output_path: $output_path"

. ~/miniconda3/etc/profile.d/conda.sh
conda activate mae 

for config in mask07
do
    echo "pre-traning ${config}"
    python /Midgard/home/rickym/reproduce-mae/main.py --pretrain \
        --config "/Midgard/home/rickym/reproduce-mae/configs/masking_coco/${config}_pretrain.yaml" \
        --id "${SLURM_JOB_ID}_${config}" \
        --epochs "$epochs_pretrain" \
        --checkpoint-frequency "$checkpoint_frequency" \
        --log_image_interval "$log_image_ingerval" \
        --data-path "$data_path" \
        --output-path "$output_path" \

    echo "finetuning ${config}"
    python /Midgard/home/rickym/reproduce-mae/main.py --finetune \
        --config "configs/masking_coco/${config}_finetune.yaml" \
        --checkpoint "$output_path/checkpoints/${SLURM_JOB_ID}_${config}_pretrain/current_best.pth" \
        --epochs "$epochs_finetune" \
        --id "${SLURM_JOB_ID}_${config}" \
        --checkpoint-frequency "$checkpoint_frequency" \

    echo "test classification ${config}"
    python /Midgard/home/rickym/reproduce-mae/main.py --test-classification \
        --checkpoint "$output_path/checkpoints/${SLURM_JOB_ID}_${config}_finetune/current_best.pth" \
        --id "${SLURM_JOB_ID}_${config}" \

    echo "Finished ${config}"
done