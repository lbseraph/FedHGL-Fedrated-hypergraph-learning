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
    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.4)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--dname', default='cora')
    parser.add_argument('--method', default='FedHGN')
    parser.add_argument('--local_step', default=3, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1, 2, 3], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--num_layers2', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--hiddens_num', default=16,
                        type=int)  # Encoder hidden units

    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
    # FL setting
    # only use local data
    parser.add_argument('--local', action='store_true')
    # hyperedge completion
    parser.add_argument('--HC', action='store_true')


    # client num
    parser.add_argument('--n_client', default=5, type=int)
    # global round
    parser.add_argument('--global_rounds', default=200, type=int)
    # data distribution
    parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)
    # act as safety mode
    parser.add_argument('--safety', action='store_true')
    parser.add_argument("--epsilon", default=10, type=float)

    args = parser.parse_args()

    # put things to device
    if args.cuda in [0, 1, 2, 3]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
        # torch.cuda.memory._record_memory_history(max_entries=100000)
    else:
        device = torch.device('cpu')
    
    ### Load and preprocess data ###
    set_seed(2025)
    torch.cuda.manual_seed(2025)

    features_origin, edge_list, labels, num_vertices, GHG = load_dataset(args, device)

    print("Begin Train!")
    ### Training loop ###
    runtime_list = []
    Final_test_accuracy = []
    avg_train_loss = []
    avg_test_accuracy = []
    for run in tqdm(range(args.runs)):
        # According to the communication range, obtain the subgraph and the extended graph consisting of the neighbors of all nodes in the subgraph
        features = features_origin.clone()
        split_X, split_Y, split_structure, split_train_mask, split_val_mask, split_test_mask = split_dataset(features, edge_list, labels, num_vertices, GHG, args, device)
        # Create a new federated learning client and server
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
        
        server = Server(clients, device, args) 
        
        start_time = time.time()
        for i in range(args.global_rounds):
            server.train(i)
            # print("round", i, torch.cuda.memory_allocated(), torch.cuda.memory_cached())
        end_time = time.time()
        runtime_list.append(end_time - start_time)
        
        train_data_weights = [torch.sum(train_mask) for train_mask in split_train_mask]
        test_data_weights = [torch.sum(test_mask) for test_mask in split_test_mask]
        val_data_weights = [torch.sum(val_mask) for val_mask in split_val_mask]

        test_result = []
        for client in server.clients:
            if args.safety:
                client.model.load_state_dict(torch.load(f"model/{type(client.model).__name__}_client_{args.n_client}_{client.rank}_{args.epsilon}.pt"))
            else:
                client.model.load_state_dict(torch.load(f"model/{type(client.model).__name__}_client_{args.n_client}_{client.rank}.pt"))

        test_results = np.array([client.local_test() for client in server.clients])

        average_test_loss = np.average(
            [row[0] for row in test_results], weights=test_data_weights, axis=0
        )
        average_test_accuracy = np.average(
            [row[1] for row in test_results], weights=test_data_weights, axis=0
        )

        Final_test_accuracy.append(average_test_accuracy)
        print("loss", average_test_loss, "acc", average_test_accuracy) 


        results = np.array([clients.get_all_loss_accuracy() for clients in server.clients])
        
        round_train_loss = np.average(
            [row[0] for row in results], weights=train_data_weights, axis=0
        )
        round_test_accuracy = np.average(
            [row[1] for row in results], weights=test_data_weights, axis=0
        )
        avg_train_loss.append(round_train_loss)
        avg_test_accuracy.append(round_test_accuracy)

    print("Final Test Accuracy: ", round(np.mean(Final_test_accuracy), 4), round(np.std(Final_test_accuracy), 4))
    
    # Use numpy.stack to stack these arrays along a new axis
    stacked_train_loss = np.stack(avg_train_loss, axis=0) 
    stacked_test_accuracy = np.stack(avg_test_accuracy, axis=0)

    # Computes the mean of the stacked array along the axis of stacking
    mean_train_loss = np.mean(stacked_train_loss, axis=0)
    mean_test_accuracy = np.mean(stacked_test_accuracy, axis=0)
    print(mean_train_loss, mean_test_accuracy)
  