数据集：
解压至data文件夹

运行：
python ./src/main.py --dname cora --layers_num 2 --hiddens_num 16 --n_client 2 --local_step 5 --runs 20 --cuda 0 --lr 0.01 --add_self_loop --method FedHGN --global_rounds 200 --local

错误：
:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cuda\ScatterGatherKernel.cu:145: block: [16,0,0], thread: [0,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
邻接矩阵超范围