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
from torch_geometric.datasets import Planetoid, ModelNet

def extract_subgraph(edge_list, idx_list):
    # 创建一个映射，将idx_list中的节点映射到新的编号
    old_to_new = {node: i for i, node in enumerate(idx_list)}
    new_edge_list = []  # 新图的边集
    # 遍历原图中的每条边
    for edge in edge_list:
        # 如果所有节点都在idx_list中，则添加到新图中
        if all(node in old_to_new for node in edge):
            # 将旧节点编号映射到新编号
            new_edge_list.append(tuple(old_to_new[node] for node in edge))
    
    return new_edge_list

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
    
    return new_edge_list, len(included_nodes), list(neighbors)

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
