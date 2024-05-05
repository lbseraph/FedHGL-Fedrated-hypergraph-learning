#!/usr/bin/env python
# coding: utf-8

import os
import time
# import math
import torch
# import pickle
import argparse

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from models import *
from preprocessing import *
from client import Client
from server import Server
from data_loader import dataset_Hypergraph

def load_dataset(args):
    existing_dataset = ['ModelNet40', 'NTU2012',
                    'cora', 'citeseer', 'pubmed']
    if args.dname not in existing_dataset:
        raise RuntimeError("Unknown dataset!")
        
    simple_graph_method = ["FedGCN", "FedSage+"]
    hypergraph_method = ["FedHGN"]
    if args.method in hypergraph_method:
        # 读取超图数据集
        dname = args.dname
        if dname in ['cora', 'citeseer','pubmed']:
            p2raw = './data/AllSet_all_raw_data/cocitation/'
        else:
            p2raw = './data/AllSet_all_raw_data/'
        dataset = dataset_Hypergraph(name=dname,root = './data/pyg_data/hypergraph_dataset_updated/',
                                        p2raw = p2raw)
        data = dataset.data
        # print(data.edge_index)
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes

        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id == consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
        edge_index = data.edge_index
        num_nodes = data.n_x
        if args.add_self_loop:
            data.edge_index = Add_Self_Loops(edge_index, num_nodes)
        data = ExtractV2E(data)
        
        # 联邦学习根据客户端划分节点
        split_idx = label_dirichlet_partition(
            data.y, len(data.y), args.num_classes, args.n_client, args.iid_beta, device
        )
        # 随机划分训练集测试集验证集

        split_idx = rand_train_test_idx(split_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)
        if not args.local:
            split_idx = get_in_comm_indexes(data, split_idx, args.safty)
        x_clients = [data.x[split_idx[i]["total"]] for i in range(len(split_idx))]
        y_clients = [data.y[split_idx[i]["total"]] for i in range(len(split_idx))]        
        # 根据节点id计算关系矩阵
        H_clients = ConstructH(data, split_idx)
        # print(H_clients)
        # 计算拉普拉斯矩阵
        G_clients = generate_G_from_H(H_clients)

        return split_idx, G_clients, x_clients, y_clients
    elif args.method in simple_graph_method: 
        # 读取简单图数据集
        features, adj, labels, idx_train, idx_val, idx_test = load_simple_graph(args.dname)
        args.num_classes = labels.max().item() + 1
        row, col, _ = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        if args.add_self_loop:
            edge_index = Add_Self_Loops(edge_index, len(labels))
        edge_index = edge_index.to(device)
        split_data_indexes = label_dirichlet_partition(
            labels, len(labels), args.num_classes, args.n_client, args.iid_beta, device
        )
        # print(len(split_data_indexes))
        split_idx, edge_indexes_clients = get_simple_in_comm_indexes(
            edge_index, split_data_indexes, args.n_client, 1, idx_train, idx_val, idx_test,
        )        
        
        args.num_features = features.shape[1]
        args.num_classes = labels.max().item() + 1
        x_clients = [features[split_idx[i]["total"]] for i in range(len(split_idx))]
        y_clients = [labels[split_idx[i]["total"]] for i in range(len(split_idx))]
        return split_idx, edge_indexes_clients, x_clients, y_clients
    else:
        raise RuntimeError("Unknown method!")  
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')
    parser.add_argument('--method', default='FedHGN')
    parser.add_argument('--local_step', default=3, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--layers_num', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--hiddens_num', default=64,
                        type=int)  # Encoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)

    parser.add_argument('--add_self_loop', action='store_true')

    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
    # FL setting
    # only use local data
    parser.add_argument('--local', action='store_true')
    # client num
    parser.add_argument('--n_client', default=5, type=int)
    # global round
    parser.add_argument('--global_rounds', default=200, type=int)
    # data distribution
    parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)
    # act as safty mode
    parser.add_argument('--safty', action='store_true')
    #     Use the line below for .py file
    args = parser.parse_args()
    
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    ### Load and preprocess data ###
    np.random.seed(12)
    torch.manual_seed(12)
    
    split_idx, HorA, x_clients, y_clients = load_dataset(args)
    print("Begin Train!")
    ### Training loop ###
    runtime_list = []
    for run in tqdm(range(args.runs)):
        # 根据通信范围，获取子图和子图所有节点的邻居构成的扩充图
        
        # 新建联邦学习客户端和服务器
        clients = [
                Client(
                    rank = i,
                    G = HorA[i],
                    features = x_clients[i],
                    labels=y_clients[i],
                    idx=split_idx[i],
                    device=device,
                    args=args,                    
                )
                for i in range(args.n_client)
            ]
        torch.cuda.empty_cache()
        server = Server(clients, device, args) 
        
        start_time = time.time()
        for i in range(args.global_rounds):
            server.train(i)
        end_time = time.time()
        runtime_list.append(end_time - start_time)
        
        results = np.array([clients.get_all_loss_accuray() for clients in server.clients])

        train_data_weights = [len(node_ids["train"]) for node_ids in split_idx]
        test_data_weights = [len(node_ids["test"]) for node_ids in split_idx]
        val_data_weights = [len(node_ids["val"]) for node_ids in split_idx]
        
        average_train_loss = np.average(
            [row[0] for row in results], weights=train_data_weights, axis=0
        )
        average_train_accuracy = np.average(
            [row[1] for row in results], weights=train_data_weights, axis=0
        )
        average_test_loss = np.average(
            [row[2] for row in results], weights=test_data_weights, axis=0
        )
        average_test_accuracy = np.average(
            [row[3] for row in results], weights=test_data_weights, axis=0
        )
        average_val_loss = np.average(
            [row[4] for row in results], weights=val_data_weights, axis=0
        )
        average_val_accuracy = np.average(
            [row[5] for row in results], weights=val_data_weights, axis=0
        )

        test_results = np.array([client.local_test() for client in server.clients])
        val_results = np.array([client.local_val() for client in server.clients])

        average_final_val_loss = np.average(
            [row[0] for row in val_results], weights=test_data_weights, axis=0
        )
        average_final_val_accuracy = np.average(
            [row[1] for row in val_results], weights=test_data_weights, axis=0
        )

        average_final_test_loss = np.average(
            [row[0] for row in test_results], weights=val_data_weights, axis=0
        )
        average_final_test_accuracy = np.average(
            [row[1] for row in test_results], weights=val_data_weights, axis=0
        )

        print("val", average_final_val_loss, average_final_val_accuracy)
        print("test", average_final_test_loss, average_final_test_accuracy)    
  