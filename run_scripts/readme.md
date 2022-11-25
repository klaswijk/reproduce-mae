sbatch --export id_backup=337605/1300, dataset=fashion_mnist,LR=0.0001,S=5,K=1,max_epochs=10000,look_ahead=50,batch_size=100,warmup=100,orgmis=0,lambda=0,seed=100,test_bs=20,test_k=5000 ./run_scripts/run_train_test.sbatch


dataset path: /local_storage/datasets
output path: /local_storage/users/rickym/models
code path: /Midgard/home/rickym/reproduce-mae
