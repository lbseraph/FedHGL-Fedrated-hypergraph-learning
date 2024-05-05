数据集：
解压至data文件夹

运行：
python ./src/main.py --method FedGCN --dname cora --layers_num 2 --hiddens_num 16 --n_client 10 --local_step 3 --runs 10 --cuda 0 --lr 0.001 --add_self_loop

python ./src/main.py --method FedGCN --dname pubmed --layers_num 2 --hiddens_num 16 --n_client 5 --local_step 3 --runs 10 --cuda 0 --lr 0.001 --add_self_loop

python ./src/main.py --method FedHGN --dname cora --layers_num 2 --hiddens_num 16 --n_client 5 --local_step 3 --runs 10 --cuda 0 --lr 0.001 --add_self_loop