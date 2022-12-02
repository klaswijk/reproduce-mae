

###########
# run test
###########
# imagenette 
sbatch --export epochs_pretrain=100,epochs_finetune=100,checkpoint_frequency=100,log_image_ingerval=2,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models,config=imagenette_testing /Midgard/home/rickym/reproduce-mae/run_scripts/run_full.sbatch

# coco 
sbatch --export epochs_pretrain=100,epochs_finetune=100,checkpoint_frequency=100,log_image_ingerval=2,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models,config=coco_testing /Midgard/home/rickym/reproduce-mae/run_scripts/run_full.sbatch

###########
# run real
###########
# imagenette 346452
sbatch --export epochs_pretrain=4000,epochs_finetune=4000,checkpoint_frequency=500,log_image_ingerval=50,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models,config=imagenette /Midgard/home/rickym/reproduce-mae/run_scripts/run_full.sbatch

# coco 346451
sbatch --export epochs_pretrain=4000,epochs_finetune=4000,checkpoint_frequency=500,log_image_ingerval=50,data_path=/local_storage/datasets/rickym/,output_path=/local_storage/users/rickym/models,config=coco /Midgard/home/rickym/reproduce-mae/run_scripts/run_full.sbatch

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