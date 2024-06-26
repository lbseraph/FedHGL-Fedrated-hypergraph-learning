import sys
import numpy as np
import torch
import pickle as pkl
import math
import networkx as nx
import scipy.sparse as sp

from collections import Counter
import torch_geometric
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, Cooking200, News20

# 读取数据集
def load_dataset(device, args):
       
    simple_graph_method = ["FedGCN", "FedSage"]
    hypergraph_method = ["FedHGN"]
    
    cite_dataset = ["cora", "pubmed", "citeseer"]
    hypergraph_dataset = ["cooking", "news"]
    
    if args.dname == "cora":
        data = Cora() 
    elif args.dname == "pubmed":
        data = Pubmed()
    elif args.dname == "citeseer":
        data = Citeseer()
    elif args.dname == "cooking":
        data = Cooking200()
    elif args.dname == "news":
        data = News20()
        
    if args.dname in cite_dataset:
        args.num_features = data["dim_features"]
        features = data["features"]
        if args.dname in ["cora", "citeseer"]:
            # 修正数据集，取消归一化
            # 查找小于0或大于1的元素  
            mask = torch.gt(features, 0) & torch.lt(features, 1.1)
            # # 使用布尔索引选出满足条件的元素
            features[mask] = 1.0
    elif args.dname == "cooking":
        features = torch.eye(data["num_vertices"])
        args.num_features = features.shape[1]
    elif args.dname == "news":
        args.num_features = data["dim_features"]
        features = data["features"]
           
    args.num_classes = data["num_classes"]
    split_idx = label_dirichlet_partition(
        data["labels"], data["num_vertices"], args.num_classes, args.n_client, args.iid_beta, device
    )

    split_structure = []
    split_train_mask = []
    split_val_mask = []
    split_test_mask = []    

    edge_list = data["edge_list"]
    if args.dname in cite_dataset and args.method in hypergraph_method:
        # print(data["num_vertices"])
        G = Graph(data["num_vertices"], data["edge_list"])
        HG = Hypergraph.from_graph_kHop(G, k=1)
        edge_list = HG.e_of_group("main")[0]
    if args.dname in hypergraph_dataset and args.method in simple_graph_method:
        HG = Hypergraph(data["num_vertices"], data["edge_list"])
        G = Graph.from_hypergraph_clique(HG, weighted=True)
        edge_list = G.e[0] 
    if args.dname in hypergraph_dataset and args.method in hypergraph_method:
        HG = Hypergraph(data["num_vertices"], data["edge_list"])

    # pre-train process(first layer)
    if args.method == "FedHGN" and not args.local:
        old_split_X = [features[split_idx[i]] for i in range(args.n_client)]
        features = HG.smoothing_with_HGNN(features)

    split_X = [features[split_idx[i]] for i in range(args.n_client)]
    split_Y = [data["labels"][split_idx[i]] for i in range(args.n_client)]    

    for i in range(args.n_client):
        
        node_num = len(split_idx[i])
        train_mask, test_mask, val_mask = rand_train_test_idx(node_num, args.train_prop, args.valid_prop) 
        if args.method in simple_graph_method:
            if args.local:
                new_edge_list = extract_subgraph(edge_list, split_idx[i])
            else:

                new_edge_list, neighbors = extract_subgraph_with_neighbors(edge_list, split_idx[i])
                node_num = node_num + len(neighbors)
                split_X[i] = torch.cat([split_X[i], features[neighbors]], dim=0)
                split_Y[i] = torch.cat([split_Y[i], data["labels"][neighbors]], dim=0)
                for _ in range(args.num_neighbor - 1):
                    new_edge_list, neighbors = extract_subgraph_with_neighbors(edge_list, split_idx[i] + neighbors)
                    node_num = node_num + len(neighbors)
                    split_X[i] = torch.cat([split_X[i], features[neighbors]], dim=0)
                    split_Y[i] = torch.cat([split_Y[i], data["labels"][neighbors]], dim=0)
                
                if args.method == "FedSage":
                    G_noise = np.random.normal(loc=0, scale = 0.1, size=features[neighbors].shape).astype(np.float32)
                    features[neighbors] += G_noise
                split_structure.append(Graph(num_v=node_num, e_list=new_edge_list).A)

        elif args.method == "FedHGN":
            new_edge_list = extract_subgraph(edge_list, split_idx[i])
            if args.local:
                HG = Hypergraph(num_v=node_num, e_list=new_edge_list)
                # pre-training process
                split_X[i] = HG.smoothing_with_HGNN(split_X[i])
                for _ in range(args.num_layers - 1):
                    split_X[i] = HG.smoothing_with_HGNN(split_X[i])
            else:
                # pre-train process(second layer)
                HG1 = Hypergraph(num_v=node_num, e_list=new_edge_list)
                features[split_idx[i]] = HG1.smoothing_with_HGNN(old_split_X[i])
                new_edge_list, neighbors = extract_subgraph_with_neighbors(edge_list, split_idx[i])
                HG = Hypergraph(num_v=node_num + len(neighbors), e_list=new_edge_list)
                split_X[i] = torch.cat([split_X[i], features[neighbors]], dim=0)
                split_Y[i] = torch.cat([split_Y[i], data["labels"][neighbors]], dim=0)
                split_X[i] = HG.smoothing_with_HGNN(split_X[i])

            split_structure.append(HG)
            

        split_train_mask.append(train_mask)
        split_val_mask.append(val_mask)
        split_test_mask.append(test_mask)

    return split_X, split_Y, split_structure, split_train_mask, split_val_mask, split_test_mask

# 提取仅包含客户端自身节点的子图
def extract_subgraph(edge_list, idx_list):
    # 创建一个映射，将idx_list中的节点映射到新的编号
    # print(edge_list)
    old_to_new = {node: i for i, node in enumerate(idx_list)}
    new_edge_list = []  # 新图的边集
    # 遍历原图中的每条边
    for edge in edge_list:
        # 如果所有节点都在idx_list中，则添加到新图中
        if all(node in old_to_new for node in edge):
            # 将旧节点编号映射到新编号
            new_edge_list.append(tuple(old_to_new[node] for node in edge))
        # 如果只有部分节点在idx_list中，则只添加部分节点(剩余节点大于等于2)
        elif sum(node in old_to_new for node in edge) >= 2:
            new_edge_list.append(tuple(old_to_new[node] for node in edge if node in old_to_new))
    
    return new_edge_list

# 提取包含包含客户端自身节点，加上客户端边界节点邻居节点的子图
def extract_subgraph_with_neighbors(edge_list, idx_list):

    # 将idx_list转换为集合，以便快速检查元素
    idx_set = set(idx_list)
    included_nodes = set(idx_list)
    neighbors = set()
    # 遍历所有边以找到所有邻居
    for edge in edge_list:
        if any(node in idx_set for node in edge):
            neighbors.update(edge)
    
    # 排除已经在idx_list中的节点，只保留真正的邻居节点
    neighbors.difference_update(idx_set)
    included_nodes.update(neighbors)
    
    # 创建新的节点映射，idx_list中的节点优先
    old_to_new = {node: i for i, node in enumerate(list(idx_list) + list(neighbors))}
    
    # 创建新的边集，使用新的节点编号
    new_edge_list = []
    for edge in edge_list:
        if all(node in included_nodes for node in edge):
            new_edge_list.append(tuple(old_to_new[node] for node in edge))

    return new_edge_list, list(neighbors)

# 提取
# def extract_subgraph_with_neighbors_safe(hyperedges, idx_list, node_client):
#     # 过滤超边
#     filtered_hyperedges = [
#         edge for edge in hyperedges
#         if sum(node in idx_list for node in edge) >= 2
#     ]

#     # 第二步：从超边中移除孤立的节点
#     final_hyperedges = []
#     for edge in filtered_hyperedges:
#         # 统计每个客户端中节点的数量
#         client_node_count = {}
#         for node in edge:
#             client = node_client[node]
#             if client not in client_node_count:
#                 client_node_count[client] = 0
#             client_node_count[client] += 1
        
#         # 只保留那些客户端中有至少两个节点的节点
#         new_edge = [node for node in edge if client_node_count[node_client[node]] >= 2]
#         if len(new_edge) >= 2:
#             final_hyperedges.append(new_edge)


#     # 创建新的节点映射
#     all_nodes = set(node for edge in final_hyperedges for node in edge)
#         # 新节点数和新邻居
#     new_nodes = sorted(set(all_nodes) - set(idx_list))

#     old_to_new = {node: i for i, node in enumerate(list(idx_list) + new_nodes)}

#     # 使用新映射重新编号超边
#     remapped_hyperedges = [[old_to_new[node] for node in edge] for edge in final_hyperedges]

#     return remapped_hyperedges, list(new_nodes)

def label_dirichlet_partition(labels, N: int, K: int, n_parties: int, beta: float, device):
    """
    This function partitions data based on labels by using the Dirichlet distribution, to ensure even distribution of samples

    Arguments:
    labels: (NumPy array) - An array with labels or categories for each data point
    N: (int) - Total number of data points in the dataset
    K: (int) - Total number of unique labels
    n_parties: (int) - The number of groups into which the data should be partitioned
    beta: (float) - Dirichlet distribution parameter value

    Return:
    split_data_indexes (list) - list indices of data points assigned into groups

    """
    min_size = 0
    min_require_size = 10

    split_data_indexes = []

    while min_size < min_require_size:
        idx_batch: list[list[int]] = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(labels.cpu() == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))

            proportions = np.array(
                [
                    p * (len(idx_j) < N / n_parties)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )

            proportions = proportions / proportions.sum()

            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        idx_batch[j].sort()
        split_data_indexes.append(idx_batch[j])
        
    return split_data_indexes

def generate_bool_tensor(length, true_count, mask=None):
    if mask is not None and len(mask) != length:
        raise ValueError("Mask length must be the same as the tensor length")
    
    if mask is not None and true_count > (length - mask.sum()):
        raise ValueError("Number of True values requested exceeds available False positions in the mask")
    
    # Create a tensor of the specified length filled with False
    tensor = torch.zeros(length, dtype=torch.bool)
    
    if mask is None:
        mask = torch.zeros(length, dtype=torch.bool)
    
    # Available indices where we can place True (those not True in mask)
    available_indices = torch.where(~mask)[0]
    
    # Randomly choose indices from available indices to set to True
    true_indices = available_indices[torch.randperm(len(available_indices))[:true_count]]
    tensor[true_indices] = True
    
    return tensor

def rand_train_test_idx(node_num, train_porb, val_prob):

    trainCount = node_num * train_porb
    valCount = node_num * val_prob
    train_mask = generate_bool_tensor(node_num, int(trainCount))
    val_mask = generate_bool_tensor(node_num, int(valCount), train_mask)
    test_mask = ~(train_mask | val_mask)
    return train_mask, val_mask, test_mask

