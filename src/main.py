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

from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, Cooking200, MovieLens1M
from dhg.random import set_seed

# 读取数据集
def load_dataset(args):
       
    simple_graph_method = ["FedGCN", "FedSage+"]
    hypergraph_method = ["FedHGN"]
    
    cite_dataset = ["cora", "pubmed", "citeseer"]
    hypergraph_dataset = ["cooking"]
    
    if args.dname == "cora":
        data = Cora() 
    elif args.dname == "pubmed":
        data = Pubmed()
    elif args.dname == "citeseer":
        data = Citeseer()
    elif args.dname == "cooking":
        data = Cooking200()
    elif args.dname == "movieLens":
        data = MovieLens1M()
        
    if args.dname in cite_dataset:
        args.num_features = data["dim_features"]
        features = data["features"]
    elif args.dname in hypergraph_dataset:
        features = torch.eye(data["num_vertices"])
        args.num_features = features.shape[1]
           
    args.num_classes = data["num_classes"]
    split_idx = label_dirichlet_partition(
        data["labels"], data["num_vertices"], args.num_classes, args.n_client, args.iid_beta, device
    )
    split_X = [features[split_idx[i]] for i in range(args.n_client)]
    split_Y = [data["labels"][split_idx[i]] for i in range(args.n_client)]        
    
    split_structrue = []
    split_train_mask = []
    split_val_mask = []
    split_test_mask = []    

    edge_list = data["edge_list"]
    if args.dname in cite_dataset:
        # print(data["num_vertices"])
        if args.method in hypergraph_method:
            G = Graph(data["num_vertices"], data["edge_list"])
            HG = Hypergraph.from_graph_kHop(G, k=1)
            edge_list = HG.e_of_group("main")[0]
    elif args.dname in hypergraph_dataset:
        if args.method in simple_graph_method:
            HG = Hypergraph(data["num_vertices"], data["edge_list"])
            G = Graph.from_hypergraph_clique(HG, weighted=True)
            edge_list = G.e[0] 

    for i in range(args.n_client):
        
        node_num = len(split_idx[i])
    
        # print("new_edge_list", new_edge_list, len(new_edge_list), len(edge_list), node_num)
        # print(data["labels"][1972] == split_Y[i][978], args.num_features)

        train_mask, test_mask, val_mask = rand_train_test_idx(node_num, args.train_prop, args.valid_prop) 
        
        if args.local:
            new_edge_list = extract_subgraph(edge_list, split_idx[i])
            # print(len(new_edge_list))
        else:
            new_edge_list, new_node_num, neigbors = extract_subgraph_with_neighbors(edge_list, split_idx[i])
            # print(len(new_edge_list))
            split_X[i] = torch.cat([split_X[i], features[neigbors]], dim=0)
            split_Y[i] = torch.cat([split_Y[i], data["labels"][neigbors]], dim=0)
            train_mask = torch.cat([train_mask, torch.zeros(new_node_num - node_num, dtype=torch.bool)], dim=0)
            test_mask = torch.cat([test_mask, torch.zeros(new_node_num - node_num, dtype=torch.bool)], dim=0)
            val_mask = torch.cat([val_mask, torch.zeros(new_node_num - node_num, dtype=torch.bool)], dim=0)
            node_num = new_node_num
        
        # print(len(new_edge_list))

        # print(torch.sum(train_mask), torch.sum(val_mask), torch.sum(test_mask))
        # exit()
        split_train_mask.append(train_mask)
        split_val_mask.append(val_mask)
        split_test_mask.append(test_mask) 
        if args.method in simple_graph_method:
            # split_structrue.append(G.A)
            split_structrue.append(Graph(num_v=node_num, e_list=new_edge_list, extra_selfloop=True).A)
        elif args.method in hypergraph_method:
            split_structrue.append(Hypergraph(num_v=node_num, e_list=new_edge_list))
            
    return split_X, split_Y, split_structrue, split_train_mask, split_val_mask, split_test_mask
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.03)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='cora')
    parser.add_argument('--method', default='FedHGN')
    parser.add_argument('--local_step', default=5, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--layers_num', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--hiddens_num', default=32,
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
    parser.add_argument('--global_rounds', default=100, type=int)
    # data distribution
    parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)
    # act as safty mode
    parser.add_argument('--safty', action='store_true')
    args = parser.parse_args()
    
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    ### Load and preprocess data ###
    set_seed(2024)
    split_X, split_Y, split_structrue, split_train_mask, split_val_mask, split_test_mask = load_dataset(args)
    
    print("Begin Train!")
    ### Training loop ###
    runtime_list = []
    for run in tqdm(range(args.runs)):
        # 根据通信范围，获取子图和子图所有节点的邻居构成的扩充图
        
        # 新建联邦学习客户端和服务器
        clients = [
                Client(
                    rank = i,
                    sturcture = split_structrue[i],
                    features = split_X[i],
                    labels=split_Y[i],
                    train_mask=split_train_mask[i],
                    val_mask=split_val_mask[i],
                    test_mask=split_test_mask[i],
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

        train_data_weights = [torch.sum(train_mask) for train_mask in split_train_mask]
        test_data_weights = [torch.sum(test_mask) for test_mask in split_test_mask]
        val_data_weights = [torch.sum(val_mask) for val_mask in split_val_mask]
        
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
  