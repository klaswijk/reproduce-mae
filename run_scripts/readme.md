sbatch --export epochs=100, checkpoint-frequency=10,log_image_ingerval=2, data-path=/local_storage/datasets/cifar-10-batches-py, output-path=/local_storage/users/rickym/models /Midgard/home/rickym/reproduce-mae/run_scripts/run_train_test.sbatch


dataset path: /local_storage/datasets
output path: /local_storage/users/rickym/models
code path: /Midgard/home/rickym/reproduce-mae

# example dataset path 
/local_storage/datasets/imagenette2-160
/local_storage/datasets/cifar-10-batches-py

# example 


# inspect results 
salloc --gres=gpu:0 --mem=1GB --cpus-per-task=1 --constrain=smaug --time=1:00:00