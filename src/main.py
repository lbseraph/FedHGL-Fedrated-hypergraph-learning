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

from dhg.random import set_seed
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.03)
    parser.add_argument('--valid_prop', type=float, default=0.2)
    parser.add_argument('--dname', default='cora')
    parser.add_argument('--method', default='FedHGN')
    parser.add_argument('--local_step', default=3, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--hiddens_num', default=16,
                        type=int)  # Encoder hidden units

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
    # act as safety mode
    parser.add_argument('--safety', action='store_true')
    parser.add_argument('--num_neighbor', default=1, type=int)  # Placeholder

    args = parser.parse_args()
    
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    ### Load and preprocess data ###
    set_seed(2025)
    split_X, split_Y, split_structure, split_train_mask, split_val_mask, split_test_mask = load_dataset(device, args)
    
    print("Begin Train!")
    ### Training loop ###
    runtime_list = []
    Final_test_accuracy = []
    for run in tqdm(range(args.runs)):
        # 根据通信范围，获取子图和子图所有节点的邻居构成的扩充图

        # 新建联邦学习客户端和服务器
        clients = [
                Client(
                    rank = i,
                    structure = split_structure[i],
                    features = split_X[i],
                    labels = split_Y[i],
                    train_mask = split_train_mask[i],
                    val_mask = split_val_mask[i],
                    test_mask = split_test_mask[i],
                    device = device,
                    args = args,
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
        
        results = np.array([clients.get_all_loss_accuracy() for clients in server.clients])

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

        Final_test_accuracy.append(average_final_test_accuracy)

        print("val", average_final_val_loss, average_final_val_accuracy)
        print("test", average_final_test_loss, average_final_test_accuracy) 
    print("Final Test Accuracy: ", np.mean(Final_test_accuracy), np.std(Final_test_accuracy))
  