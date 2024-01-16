#!/usr/bin/env python
# coding: utf-8

import os
import time
# import math
import torch
# import pickle
import argparse

import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from layers import *
from models import *
from preprocessing import *
from client import Client
from server import Server
from data_loader import dataset_Hypergraph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')
    parser.add_argument('--method', default='HGNN')
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
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)

    # FL setting
    # only use local data
    parser.add_argument('--local', action='store_true')
    # client num
    parser.add_argument('--n_client', default=5, type=int)
    # global round
    parser.add_argument('--global_rounds', default=100, type=int)
    # data distribution
    parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    
    
    # # Part 1: Load data
    
    ### Load and preprocess data ###
    existing_dataset = ['ModelNet40', 'NTU2012',
                        'cora', 'citeseer', 'pubmed']
    
    
    if args.dname in existing_dataset:
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
    
    if args.method == 'HGNN':
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
    else:
        raise RuntimeError("Unknown method!")     
        
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    # # Part 3: Main. Training + Evaluation
    print("Begin Train!")
    ### Training loop ###
    runtime_list = []
    for run in tqdm(range(args.runs)):
        # 联邦学习根据客户端划分节点
        split_idx = label_dirichlet_partition(
            data.y, len(data.y), args.num_classes, args.n_client, args.iid_beta, device
        )
        # 随机划分训练集测试集验证集
        split_idx = rand_train_test_idx(split_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)
        # 根据通信范围，获取子图和子图所有节点的邻居构成的扩充图
        if not args.local:
            split_idx = get_in_comm_indexes(data, split_idx)
        # 根据节点id计算关系矩阵
        H_clients = ConstructH(data, split_idx)
        # 计算拉普拉斯矩阵
        G_clients = generate_G_from_H(H_clients)
        
        # 新建联邦学习客户端和服务器
        clients = [
                Client(
                    i,
                    G_clients[i],
                    data.y[split_idx[i]["total"]],
                    data.x[split_idx[i]["total"]],
                    split_idx[i],
                    device,
                    args,
                )
                for i in range(args.n_client)
            ]
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
        print("average_train_loss", average_train_loss, "\n",
              "average_train_accuracy", average_train_accuracy,  "\n",
              "average_test_loss", average_test_loss,  "\n",
              "average_test_accuracy", average_test_accuracy,  "\n",
              "average_val_loss", average_val_loss,  "\n",
              "average_val_accuracy", average_val_accuracy)


    
  