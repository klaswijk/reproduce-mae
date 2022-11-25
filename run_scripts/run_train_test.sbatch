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

conda activate /Midgard/home/rickym/miniconda3/envs/mae 


echo "traning"
/Midgard/home/rickym/julia-1.7.2/bin/julia /Midgard/home/rickym/julia_vae/main_train.jl ${id_backup} ${dataset} ${S} ${LR} ${batch_size} ${K} ${m
ax_epochs} ${look_ahead} ${warmup} ${orgmis} ${lambda} ${seed}

echo "testing"
/Midgard/home/rickym/julia-1.7.2/bin/julia /Midgard/home/rickym/julia_vae/main_test.jl ${id} ${dataset} ${test_bs} ${test_k} ${lambda} ${seed}

echo "Finished"
