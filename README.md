数据集：
解压至data文件夹


错误：
:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cuda\ScatterGatherKernel.cu:145: block: [16,0,0], thread: [0,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
邻接矩阵超范围

运行：
[cora]
python ./src/main.py --dname cora --num_layers 2 --num_neighbor 1 --n_client 5 --add_self_loop --method FedHGN --train_prop 0.05
Final Test Accuracy:  0.8167950922443661 0.00470751775694162

python ./src/main.py --dname cora --num_layers 2 --num_neighbor 1 --n_client 5 --add_self_loop --method FedGCN --train_prop 0.05
Final Test Accuracy:  0.8069988611556747 0.004063316658617443

python ./src/main.py --dname cora --num_layers 2 --num_neighbor 2 --n_client 5 --add_self_loop --method FedGCN --train_prop 0.05
Final Test Accuracy:  0.8049668434898596 0.005563424679918293

python ./src/main.py --dname cora --num_layers 2 --num_neighbor 1 --n_client 5 --add_self_loop --method FedSage --train_prop 0.05
Final Test Accuracy:  0.8031223526129082 0.007075060034440406

[citeseer]
python ./src/main.py --dname citeseer --num_layers 2 --num_neighbor 1 --n_client 5 --add_self_loop --method FedHGN --train_prop 0.03
Final Test Accuracy:  0.6644490193920275 0.009108555401486224

python ./src/main.py --dname citeseer --num_layers 2 --num_neighbor 1 --n_client 5 --add_self_loop --method FedGCN --train_prop 0.03
Final Test Accuracy:  0.6314675898290086 0.007404242774273802

python ./src/main.py --dname citeseer --num_layers 2 --num_neighbor 2 --n_client 5 --add_self_loop --method FedGCN --train_prop 0.03
Final Test Accuracy:  0.638090498249259 0.00857624212854481

python ./src/main.py --dname citeseer --num_layers 2 --num_neighbor 1 --n_client 5 --add_self_loop --method FedSage --train_prop 0.03
Final Test Accuracy:  0.6554074814189331 0.009159190420282312

[cooking200]
python ./src/main.py --dname cooking --num_layers 2 --num_neighbor 1 --n_client 5 --add_self_loop --method FedHGN --train_prop 0.5 --valid_prop 0.25 --global_rounds 200
Final Test Accuracy:  0.47600016783352617 0.0004324413471549728

python ./src/main.py --dname cooking --num_layers 2 --num_neighbor 2 --n_client 1 --add_self_loop --method FedGCN --train_prop 0.5 --valid_prop 0.25 --global_rounds 200
Final Test Accuracy:  0.36102702702702705 0.018518464518493885
