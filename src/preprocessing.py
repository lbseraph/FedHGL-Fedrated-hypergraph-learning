#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""

"""
 
import torch

import numpy as np

from collections import Counter
from torch_scatter import scatter_add
from torch_geometric.nn.conv.gcn_conv import gcn_norm

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


def Add_Self_Loops(data):
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges

    if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
        print('num_hyperedges does not match! 2')
        return

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    skip_node_lst = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros(
        (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
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

def get_in_comm_indexes(data, split_idx):

    # 计算邻接
    edge_index = np.array(data.edge_index)
    
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_nodes = data.x.shape[0]
    adj = np.zeros((num_nodes, num_nodes))
    
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        for node_id in nodes_in_he:
            adj[node_id, nodes_in_he] = 1.
        cur_idx += 1
    
    for ids_client in split_idx: 
        nodes_in_com = set(ids_client["total"].tolist())
        for idx in ids_client["total"]:
            neigbor = np.where(adj[idx] == 1)[0]
            nodes_in_com.update(neigbor)
        ids_client["total"] = torch.tensor(list(nodes_in_com))
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
        
        delet_hes = set()
        for node_id in edge_index[0]:
            if node_id not in node_ids:
            # 找到要删除的元素的索引
                idx = np.where(edge_index[0] == node_id)[0]
                # 保存要删除的超边
                # print(edge_index[1][idx])
                delet_hes.update(edge_index[1][idx].tolist())
        
        num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1 - len(delet_hes)
        
        # TODO 记录跨客户端超边的数量
        # print(num_hyperedges, len(delet_hes))
        
        for he in delet_hes:
            idx = np.where(edge_index[1] == he)[0]
            edge_index = np.delete(edge_index, idx, 1)  
        
        H = np.zeros((len(node_ids), num_hyperedges))
        
        
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