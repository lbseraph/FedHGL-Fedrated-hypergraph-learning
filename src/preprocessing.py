#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""

"""
 
import sys
import numpy as np
import torch
import pickle as pkl
import torch_sparse
import networkx as nx
import scipy.sparse as sp

from collections import Counter
import torch_geometric

def Add_Self_Loops(edge_index, num_nodes):
    
    # store the nodes that already have self-loops
    skip_node_lst = []
    mask = edge_index[0, :] == edge_index[1, :]

    skip_node_lst = edge_index[0, mask]

    new_edges = torch.zeros(
        (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)

    temp_i = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][temp_i] = i
            new_edges[1][temp_i] = i
            temp_i += 1
    # print(new_edges)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return edge_index

def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    if not ((data.n_x+data.num_hyperedges-1) == data.edge_index.max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    # print(data.edge_index, cidx)
    return data

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
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(torch.tensor(idx_batch[j]))
        
    return split_data_indexes

def rand_train_test_idx(indexes, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    split_idx_list =[]
    for index in indexes:
        if not balance:
            if ignore_negative:
                labeled_nodes = torch.where(index != -1)[0]
            else:
                labeled_nodes = index

            n = labeled_nodes.shape[0]
            train_num = int(n * train_prop)
            valid_num = int(n * valid_prop)

            perm = torch.as_tensor(np.random.permutation(n))

            train_indices = perm[:train_num]
            val_indices = perm[train_num:train_num + valid_num]
            test_indices = perm[train_num + valid_num:]

            if not ignore_negative:
                return train_indices, val_indices, test_indices

            train_idx = labeled_nodes[train_indices]
            valid_idx = labeled_nodes[val_indices]
            test_idx = labeled_nodes[test_indices]

            split_idx = {'total' : index,
                        'train': train_idx,
                        'val': valid_idx,
                        'test': test_idx}
        else:
            #         ipdb.set_trace()
            indices = []
            for i in range(index.max()+1):
                index = torch.where((index == i))[0].view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)

            percls_trn = int(train_prop/(index.max()+1)*len(index))
            val_lb = int(valid_prop*len(index))
            train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
            rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
            rest_index = rest_index[torch.randperm(rest_index.size(0))]
            valid_idx = rest_index[:val_lb]
            test_idx = rest_index[val_lb:]
            split_idx = {'total' : index,
                        'train': train_idx,
                        'val': valid_idx,
                        'test': test_idx}
        split_idx_list.append(split_idx)
    return split_idx_list

def get_in_comm_indexes(data, split_idx, safty=False):

    # 计算邻接
    edge_index = np.array(data.edge_index)
    # 节点所属客户端
    nodes_client = np.zeros(len(data.y))
    if safty:
        for i in range(len(split_idx)):
            nodes_client[split_idx[i]["total"].numpy()] = i   
            safty_node = set()
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_nodes = data.x.shape[0]
    adj = np.zeros((num_nodes, num_nodes))
    
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        for node_id in nodes_in_he:
            adj[node_id, nodes_in_he] = 1.
            # find safty node which has inter neighbor
            if safty:
                for node2 in nodes_in_he:
                    if nodes_client[node_id] == nodes_client[node2] and node_id != node2:
                        safty_node.add(node_id)
            
        cur_idx += 1
    
    for ids_client in split_idx: 
        # print("begin", len(ids_client["total"]), safty)
        id_list = ids_client["total"].tolist()
        nodes_in_com = set(id_list)
        for idx in id_list:
            neighbors = np.where(adj[idx] == 1)[0]
            if safty:
                safty_neighbors = []
                safty_neighbors.append(idx)
                for neighbor in neighbors:
                    if neighbor in safty_node:
                        safty_neighbors.append(neighbor)
                # TODO delete neigbors which have no inter neigbor
                neighbors = safty_neighbors
            # print(neighbors)
            nodes_in_com.update(neighbors)
        # print(nodes_in_com)
        ids_client["total"] = torch.tensor(list(nodes_in_com))
        # print("after", len(ids_client["total"]))
    return split_idx
    
def ConstructH(data, split_idxs):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
#     ipdb.set_trace()
    H_clients = []
    for split_idx in split_idxs:
        edge_index = np.array(data.edge_index)
        node_ids = np.array(split_idx["total"])
        
        delet_nodes = np.array([], dtype=int)
        
        for node_id in edge_index[0]:
            if node_id not in node_ids:
            # 找到要删除的元素的索引
                idx = np.where(edge_index[0] == node_id)[0]
                
                delet_nodes = np.concatenate((delet_nodes, idx))
                # print(delet_nodes)
        # print(delet_nodes)
        edge_index = np.delete(edge_index, delet_nodes, 1)  
        
        # TODO 记录跨客户端超边的数量
        # print(num_hyperedges, len(delet_hes))
        
        H = np.zeros((len(node_ids), len(np.unique(edge_index[1]))))
        
        # print(edge_index)         

        cur_idx = 0
        for he in np.unique(edge_index[1]):
            nodes_in_he = edge_index[0][edge_index[1] == he]
            for node in nodes_in_he:
                H[node_ids == node, cur_idx] = 1.
            cur_idx += 1
        H_clients.append(H)
        
    return H_clients

def Cal_adj(data):
    # cal adj
    num_nodes = data.x.shape[0]
    data.adj = np.zeros((num_nodes, num_nodes))
    for he in data.edge_index:
        pass

def generate_G_from_H(H_clients):
    """
    This function generate the propagation matrix G for HGNN from incidence matrix H.
    Here we assume data.edge_index is already the incidence matrix H. (can be done by ConstructH())
    Adapted from HGNN github repo: https://github.com/iMoonLab/HGNN
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
#     ipdb.set_trace()

    G_clients = []
    for H in H_clients:
        H = np.array(H)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    #     replace nan with 0. This is caused by isolated nodes
        DV2 = np.nan_to_num(DV2)
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        G = DV2 * H * W * invDE * HT * DV2
        G_clients.append(torch.Tensor(G))
    return G_clients

def load_simple_graph(dataset_str: str):
    """
    This function loads input data from gcn/data directory

    Argument:
    dataset_str: Dataset name

    Return:
    All data input files loaded (as well as the training/test data).

    Note:
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.
    """

    if dataset_str in ["cora", "citeseer", "pubmed"]:
        names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
        objects = []
        for i in range(len(names)):
            with open("data/simple/ind.{}.{}".format(dataset_str, names[i]), "rb") as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding="latin1"))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "data/simple/ind.{}.test.index".format(dataset_str)
        )
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == "citeseer":
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1
            )
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        # print(adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = torch.LongTensor(test_idx_range.tolist())
        idx_train = torch.LongTensor(range(len(y)))
        idx_val = torch.LongTensor(range(len(y), len(y) + 500))
        # print(idx_test)
        # features = normalize(features)
        # adj = normalize(adj)    # no normalize adj here, normalize it in the training process

        features = torch.tensor(features.toarray()).float()
        adj = torch.tensor(adj.toarray()).float()
        adj = torch_sparse.tensor.SparseTensor.from_dense(adj)

        labels = torch.tensor(labels)
        labels = torch.argmax(labels, dim=1)

    return features.float(), adj, labels, idx_train, idx_val, idx_test

def parse_index_file(filename: str):
    """
    This function reads and parses an index file

    Args:
    filename: (str) - name or path of the file to parse

    Return:
    index: (list) - list of integers, each integer in the list represents int of the lines lines of the input file.
    """
    
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def get_simple_in_comm_indexes(
    edge_index: torch.Tensor,
    split_data_indexes: list,
    num_clients: int,
    L_hop: int,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
):
    """
    This function is used to extract and preprocess data indices and edge information

    Arguments:
    edge_index: (PyTorch tensor) - Edge information (connection between nodes) of the graph dataset
    split_data_indexes: (List) - A list of indices of data points assigned to a particular group post data partition
    num_clients: (int) - Total number of clients
    L_hop: (int) - Number of hops
    idx_train: (PyTorch tensor) - Indices of training data
    idx_test: (PyTorch tensor) - Indices of test data

    Returns:
    communicate_indexes: (list) - A list of indices assigned to a particular client
    in_com_train_data_indexes: (list) - A list of tensors where each tensor contains the indices of training data points available to each client
    edge_indexes_clients: (list) - A list of edge tensors representing the edges between nodes within each client's subgraph
    """
    communicate_indexes = []
    edge_indexes_clients = []
    split_idx_list = []
    
    for i in range(num_clients):
        communicate_index = split_data_indexes[i]
        
        if L_hop == 0:
            (
                communicate_index,
                current_edge_index,
                _,
                __,
            ) = torch_geometric.utils.k_hop_subgraph(
                communicate_index, 0, edge_index, relabel_nodes=True
            )
            del _
            del __

        for hop in range(L_hop):
            # print(len(communicate_index))
            if hop != L_hop - 1:
                communicate_index = torch_geometric.utils.k_hop_subgraph(
                    communicate_index, 1, edge_index, relabel_nodes=True
                )[0]
            else:
                (
                    communicate_index,
                    current_edge_index,
                    _,
                    __,
                ) = torch_geometric.utils.k_hop_subgraph(
                    node_idx=communicate_index, num_hops=1, edge_index=edge_index, relabel_nodes=True
                )
                del _
                del __
        # print(len(communicate_index))
        communicate_index = communicate_index.to("cpu")
        current_edge_index = current_edge_index.to("cpu")
        communicate_indexes.append(communicate_index)

        current_edge_index = torch_sparse.SparseTensor(
            row=current_edge_index[0],
            col=current_edge_index[1],
            sparse_sizes=(len(communicate_index), len(communicate_index)),
        )

        edge_indexes_clients.append(current_edge_index)
        split_idx_list.append({
            'total' : communicate_indexes[i],
            'train': torch.searchsorted(communicate_indexes[i], intersect1d(split_data_indexes[i], idx_train)).clone(),
            'val': torch.searchsorted(communicate_indexes[i], intersect1d(split_data_indexes[i], idx_val)).clone(),
            'test': torch.searchsorted(communicate_indexes[i], intersect1d(split_data_indexes[i], idx_test)).clone(),
        })
    return split_idx_list, edge_indexes_clients
    
def intersect1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    This function concatenates the two input tensors, finding common elements between these two

    Argument:
    t1: (PyTorch tensor) - The first input tensor for the operation
    t2: (PyTorch tensor) - The second input tensor for the operation

    Return:
    intersection: (PyTorch tensor) - Intersection of the two input tensors
    """
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection