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
from dhg.data import Cora, Pubmed, Citeseer, Cooking200, News20, Yelp3k, DBLP4k

simple_graph_method = ["FedGCN", "FedSage"]
hypergraph_method = ["FedHGN"]

cite_dataset = ["cora", "pubmed", "citeseer"]
hypergraph_dataset = ["cooking", "news", "yelp", "dblp"]

# 去除重复的边和孤立的点
def clean_edge_list(edge_list):
    unique_edges = set()
    cleaned_edge_list = []

    for edge in edge_list:
        # Convert edge to a frozenset to make it hashable and to ignore order of nodes
        edge_set = frozenset(edge)
        
        # Skip edges with only one node
        if len(edge) > 1 and edge_set not in unique_edges:
            unique_edges.add(edge_set)
            cleaned_edge_list.append(edge)

    return cleaned_edge_list

# 读取数据集
def load_dataset(args, device):
    
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
    elif args.dname == "yelp":
        data = Yelp3k()
    elif args.dname == "dblp":
        data = DBLP4k()
        
    if args.dname in cite_dataset:
        args.num_features = data["dim_features"]
        features = data["features"]
        if args.dname in ["cora", "citeseer"]:
            # 修正数据集，取消归一化
            # 查找小于0或大于1的元素  
            mask = torch.gt(features, 0) & torch.lt(features, 1.1)
            # # 使用布尔索引选出满足条件的元素
            features[mask] = 1.0
        edge_list = data["edge_list"]
    elif args.dname == "cooking":
        features = torch.eye(data["num_vertices"])
        args.num_features = features.shape[1]
        edge_list = data["edge_list"]
    elif args.dname in ["news", "yelp"]:
        args.num_features = data["dim_features"]
        features = data["features"]
        edge_list = data["edge_list"]
    elif args.dname == "dblp":
        args.num_features = data["dim_features"]
        features = data["features"]
        edge_list = data["edge_by_paper"] + data["edge_by_term"] + data["edge_by_conf"]
        # print(edge_list)
    print("hyperedge", len(edge_list))
    print("before", len(edge_list))
    edge_list = clean_edge_list(edge_list)
    print("after", len(edge_list))
    args.num_classes = data["num_classes"]

    num_vertices = data["num_vertices"]

    if args.dname in cite_dataset and args.method in hypergraph_method:
        # print(data["num_vertices"])
        G = Graph(num_vertices, edge_list)
        HG = Hypergraph.from_graph_kHop(G, k=1)
        edge_list = HG.e_of_group("main")[0]
        print("hyperedge", len(edge_list))
        # HG.add_hyperedges_from_feature_kNN(feature=features, k=4)
        # edge_list = HG.e_of_group("main")[0]
    elif args.dname in hypergraph_dataset and args.method in simple_graph_method:
        HG = Hypergraph(num_vertices, edge_list).to(device)
        G = Graph.from_hypergraph_clique(HG, weighted=True, device=device)
        edge_list = G.e[0] 
    elif args.dname in hypergraph_dataset and args.method in hypergraph_method:
        HG = Hypergraph(num_vertices, edge_list)
    else:
        HG = None

    
    print("hyperedge", len(edge_list))
    return features, edge_list, data["labels"], data["num_vertices"], HG

def find_cross_edges(edge_list, split_idx):

    # 创建一个节点到客户端的映射
    node_to_client = {}
    for client_id, nodes in enumerate(split_idx):
        for node in nodes:
            node_to_client[node] = client_id

    cross_edges = []
    for u, v in edge_list:
        # 检查 u 和 v 是否属于不同的客户端
        if node_to_client[u] != node_to_client[v]:
            cross_edges.append((u, v))

    return cross_edges

def remove_cross_edges(edge_list, sub_edge_list):

    # 将sub_edge_list转换为集合以便快速查找
    sub_edge_set = set(sub_edge_list)
    # 从edge_list中删除sub_edge_list中的边
    remaining_edges = [edge for edge in edge_list if edge not in sub_edge_set]
    return remaining_edges

# 读取数据集
def split_dataset(features, edge_list, labels, num_vertices, HG, args, device): 

    split_idx = label_dirichlet_partition(
        labels, num_vertices, args.num_classes, args.n_client, args.iid_beta, device
    )

    split_structure = []
    split_train_mask = []
    split_val_mask = []
    split_test_mask = []
    
    split_X = [features[split_idx[i]] for i in range(args.n_client)]
    split_Y = [labels[split_idx[i]] for i in range(args.n_client)]    

    # pre-train process(first layer)
    if args.method == "FedHGN" and not args.local:
        old_split_X = [features[split_idx[i]] for i in range(args.n_client)]
        features = HG.smoothing_with_HGNN(features)
    if args.method == "FedGCN" and not args.local:
        cross_edges = find_cross_edges(edge_list, split_idx)

    for i in range(args.n_client):
        
        node_num = len(split_idx[i])
        train_mask, test_mask, val_mask = rand_train_test_idx(node_num, args.train_ratio, args.val_ratio, args.test_ratio) 
        if args.method in simple_graph_method:
            if args.local:
                new_edge_list = extract_subgraph(edge_list, split_idx[i])
            else:

                new_edge_list, neighbors = extract_subgraph_with_neighbors(edge_list, split_idx[i])
                node_num = node_num + len(neighbors)

                if args.method == "FedSage":
                    
                    # if args.dname == "dblp":
                    #     scale = 0.2
                    # else:
                    scale = 0.1

                    G_noise = np.random.normal(loc=0, scale = scale, size=features.shape).astype(np.float32)
                    features += G_noise

                split_X[i] = torch.cat([split_X[i], features[neighbors]], dim=0)
                split_Y[i] = torch.cat([split_Y[i], labels[neighbors]], dim=0)

                if args.method == "FedGCN":
                    for _ in range(args.num_layers - 1):
                        # 从其它客户端处获取2跳以上邻居的信息，但是不能跨客户端
                        # print("???", len(edge_list))
                        sub_edge = remove_cross_edges(edge_list, cross_edges)
                        # print("???", len(sub_edge))
                        new_edge_list, neighbors = extract_subgraph_with_neighbors(edge_list, split_idx[i] + neighbors, sub_edge)
                        # print(len(neighbors))
                        node_num = node_num + len(neighbors)
                        split_X[i] = torch.cat([split_X[i], features[neighbors]], dim=0)
                        split_Y[i] = torch.cat([split_Y[i], labels[neighbors]], dim=0)
                
                
            split_structure.append(Graph(num_v=node_num, e_list=new_edge_list, device=device).A)

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
                split_Y[i] = torch.cat([split_Y[i], labels[neighbors]], dim=0)
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
def extract_subgraph_with_neighbors(edge_list, idx_list, sub_edge_list=None):

    # 将idx_list转换为集合，以便快速检查元素
    idx_set = set(idx_list)
    included_nodes = set(idx_list)
    neighbors = set()

    # 遍历所有边以找到所有邻居
    if sub_edge_list is None:
        for edge in edge_list:
            if any(node in idx_set for node in edge):
                neighbors.update(edge)
    else:
        for edge in sub_edge_list :
            # 不能是跨客户端边（在多层FedGCN中）
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
        if all(node in included_nodes for node in edge) and any(node in idx_set for node in edge):
            new_edge_list.append(tuple(old_to_new[node] for node in edge))

    return new_edge_list, list(neighbors)

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

def rand_train_test_idx(node_num, train_ratio, val_ratio, test_ratio):

    trainCount = node_num * train_ratio
    valCount = node_num * val_ratio
    testCount = node_num * test_ratio
    train_mask = generate_bool_tensor(node_num, int(trainCount))
    val_mask = generate_bool_tensor(node_num, int(valCount), train_mask)
    test_mask = generate_bool_tensor(node_num, int(testCount), train_mask | val_mask)
    return train_mask, val_mask, test_mask

