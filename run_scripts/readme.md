
# 345961
sbatch --export epochs=100,checkpoint_frequency=10,log_image_ingerval=2,data_path=/local_storage/datasets/,output_path=/local_storage/users/rickym/models /Midgard/home/rickym/reproduce-mae/run_scripts/run_pretraning.sbatch

# 345993
sbatch --export epochs=2000,checkpoint_frequency=100,log_image_ingerval=100,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models,config=imagenette.yaml /Midgard/home/rickym/reproduce-mae/run_scripts/run_pretraning.sbatch

# 346299
sbatch --export epochs=2000,checkpoint_frequency=500,log_image_ingerval=50,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models,config=imagenette.yaml /Midgard/home/rickym/reproduce-mae/run_scripts/run_pretraning.sbatch


###########
# run full
###########
# tiny 346311
sbatch --export epochs_pretrain=100,epochs_finetune=100,checkpoint_frequency=100,log_image_ingerval=2,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models,config=imagenette_tiny /Midgard/home/rickym/reproduce-mae/run_scripts/run_full_imagenette.sbatch

# real 
sbatch --export epochs_pretrain=4000,epochs_finetune=4000,checkpoint_frequency=500,log_image_ingerval=50,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models,config=imagenette /Midgard/home/rickym/reproduce-mae/run_scripts/run_full_imagenette.sbatch

###########

dataset path: /local_storage/datasets
output path: /local_storage/users/rickym/models
code path: /Midgard/home/rickym/reproduce-mae

# example dataset path 
/local_storage/datasets/imagenette2-160
/local_storage/datasets/cifar-10-batches-py

# example 


# inspect results 
salloc --gres=gpu:0 --mem=1GB --cpus-per-task=1 --constrain=smaug --time=1:00:00
salloc --gres=gpu:1 --mem=1GB --cpus-per-task=1 --constrain=smaug --time=1:00:00