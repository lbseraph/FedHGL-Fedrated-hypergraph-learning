import sys
import numpy as np
import torch
import pickle as pkl
import math
import networkx as nx
import scipy.sparse as sp
from torch import tensor
from collections import Counter
import torch_geometric
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, Cooking200, News20, Yelp3k, DBLP4k, IMDB4k, CoauthorshipCora, Github, Facebook

simple_graph_method = ["FedGCN", "FedSage", "FedCog"]
hypergraph_method = ["FedHGN", "HNHN", "HyperGCN"]

simple_dataset = ["cora", "pubmed", "citeseer", "github", "facebook"]
hypergraph_dataset = ["cooking", "news", "yelp", "dblp", "imdb", "cora-ca"]

# Remove duplicate edges and isolated points
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

# Reading the dataset
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
    elif args.dname == "imdb":
        data = IMDB4k()
    elif args.dname == "cora-ca":
        data = CoauthorshipCora()
    elif args.dname == "github":
        data = Github()
    elif args.dname == "facebook":
        data = Facebook()
        
    if args.dname in simple_dataset:
        if args.dname == "facebook":
            args.num_features = 4714
        else:
            args.num_features = data["dim_features"]
        features = data["features"]
        if args.dname in ["cora", "citeseer"]:
            # Correct the dataset and cancel normalization
            # Find elements less than 0 or greater than 1 
            mask = torch.gt(features, 0) & torch.lt(features, 1.1)
            # # Use Boolean indexing to select elements that meet a condition
            features[mask] = 1.0
        edge_list = data["edge_list"]
    elif args.dname == "cooking":
        features = torch.eye(data["num_vertices"])
        args.num_features = features.shape[1]
        edge_list = data["edge_list"]
    elif args.dname in ["news", "yelp", "cora-ca"]:
        args.num_features = data["dim_features"]
        features = data["features"]
        edge_list = data["edge_list"]
    elif args.dname == "dblp":
        args.num_features = data["dim_features"]
        features = data["features"]
        edge_list = data["edge_by_term"] + data["edge_by_paper"]
        # print(edge_list)
    elif args.dname == "imdb":
        args.num_features = data["dim_features"]
        features = data["features"]
        edge_list = data["edge_by_director"] + data["edge_by_actor"] # +   +data["edge_by_term"] data["edge_by_paper"]

    print("hyperedge", len(edge_list))
    print("before", len(edge_list))
    edge_list = clean_edge_list(edge_list)
    print("after", len(edge_list))
    args.num_classes = data["num_classes"]

    num_vertices = data["num_vertices"]

    if args.dname in simple_dataset and args.method in hypergraph_method:
        # print(data["num_vertices"])
        G = Graph(num_vertices, edge_list)
        HG = Hypergraph.from_graph_kHop(G, k=1)
        # edge_list = HG.e_of_group("main")[0]
        print("hyperedge", len(edge_list))
        if args.dname in ["pubmed"]:
            HG.add_hyperedges_from_feature_kNN(feature=features, k=3)
        edge_list = HG.e_of_group("main")[0]
        GHG = HG
    elif args.dname in hypergraph_dataset and args.method in simple_graph_method:
        HG = Hypergraph(num_vertices, edge_list).to(device)
        G = Graph.from_hypergraph_clique(HG, weighted=True, device=device)
        edge_list = G.e[0] 
        GHG = G
    elif args.dname in hypergraph_dataset and args.method in hypergraph_method:
        GHG = Hypergraph(num_vertices, edge_list)
    elif args.dname in simple_dataset and args.method in simple_graph_method:
        GHG = Graph(num_vertices, edge_list)
    else:
        GHG = None
    
    return features, edge_list, data["labels"], data["num_vertices"], GHG

def find_cross_edges(edge_list, split_idx):
    # Create a node-to-client mapping
    node_to_client = {}
    for client_id, nodes in enumerate(split_idx):
        for node in nodes:
            node_to_client[node] = client_id

    cross_edges = []
    for edge in edge_list:
        # Get the clients to which all nodes of the edge belong
        clients = {node_to_client[node] for node in edge}
        # If the edge spans multiple clients, add it to cross_edges
        if len(clients) > 1:
            cross_edges.append(edge)

    cross_nodes = set()
    for edge in edge_list:
        cross_nodes.update(edge)

    return cross_edges, cross_nodes

def remove_cross_edges(edge_list, sub_edge_list):

    # Convert sub_edge_list to a set for fast lookup
    sub_edge_set = set(sub_edge_list)
    # Remove the edges in sub_edge_list from edge_list
    remaining_edges = [edge for edge in edge_list if edge not in sub_edge_set]
    return remaining_edges

# Reading the dataset
def split_dataset(features, edge_list, labels, num_vertices, GHG, args, device): 

    split_idx = label_dirichlet_partition(
        labels, num_vertices, args.num_classes, args.n_client, args.iid_beta, device
    )

    split_structure = []
    split_train_mask = []
    split_val_mask = []
    split_test_mask = []
    
    if args.method == "FedHGN" and args.HC:
    #     for _ in range(args.num_layers):
    #         features = HG.smoothing_with_HGNN(features)
        new_features = GHG.smoothing_with_HGNN(features)

    split_X = [features[split_idx[i]] for i in range(args.n_client)]
    split_Y = [labels[split_idx[i]] for i in range(args.n_client)]   

    if args.method == "FedCog":
        print("before", len(edge_list))
        for i in range(args.n_client):
            HG = Hypergraph(num_v=len(split_idx[i]))
            # use knn for generate nearest neighbors
            HG.add_hyperedges_from_feature_kNN(tensor(split_X[i]), k=2)
            add_edges = HG.e_of_group("main")[0]
            safe_node = set()
            for node in split_idx[i]:
                for edge in edge_list:
                    if node in edge and all(n in split_idx[i] for n in edge):
                        safe_node.add(node)
            unsafe_node = set(split_idx[i]) - safe_node
            # print(len(split_idx[i]), len(unsafe_node),len(safe_node))
            for node in unsafe_node:
                for edge in add_edges:
                    if node in edge and all(n in split_idx[i] for n in edge):
                        edge_list.append(edge)
                        break
                        # print(add_edges)
        print("after",len(edge_list))
        GHG = Graph(num_v=num_vertices, e_list=edge_list)
        for _ in range(args.num_layers):
            features = GHG.smoothing_with_GCN(features)

    split_X = [features[split_idx[i]] for i in range(args.n_client)]
    split_Y = [labels[split_idx[i]] for i in range(args.n_client)]    

    cross_edges, cross_nodes = find_cross_edges(edge_list, split_idx)
    print("crosss", len(cross_edges), len(cross_nodes))
    if args.method == "FedSage":
        scale = 0.1
        G_noise = np.random.normal(loc=0, scale = scale, size=features.shape).astype(np.float32)
        features += G_noise

    for i in range(args.n_client):
        
        node_num = len(split_idx[i])
        train_mask, test_mask, val_mask = rand_train_test_idx(node_num, args.train_ratio, args.val_ratio, args.test_ratio) 
        split_train_mask.append(train_mask)
        split_val_mask.append(val_mask)
        split_test_mask.append(test_mask)
        if args.method in simple_graph_method:
            if not args.HC or args.method == "FedCog":
                new_edge_list = extract_subgraph(edge_list, split_idx[i])
            else:

                new_edge_list, neighbors = extract_subgraph_with_neighbors(edge_list, split_idx[i])
                node_num = node_num + len(neighbors)

                split_X[i] = torch.cat([split_X[i], features[neighbors]], dim=0)
                split_Y[i] = torch.cat([split_Y[i], labels[neighbors]], dim=0)

                if args.method == "FedGCN":
                    for _ in range(args.num_layers - 1):
                        # Get information about neighbors more than 2 hops away from other clients, but not across clients
                        # print("???", len(edge_list))
                        sub_edge = remove_cross_edges(edge_list, cross_edges)
                        # print("???", len(sub_edge))
                        new_edge_list, neighbors = extract_subgraph_with_neighbors(edge_list, split_idx[i] + neighbors, sub_edge)
                        # print(len(neighbors))
                        node_num = node_num + len(neighbors)
                        split_X[i] = torch.cat([split_X[i], features[neighbors]], dim=0)
                        split_Y[i] = torch.cat([split_Y[i], labels[neighbors]], dim=0)
                    
            split_structure.append(Graph(num_v=node_num, e_list=new_edge_list, device=device).A)

        elif args.method in hypergraph_method:

            # old_split_idx = split_idx

            if not args.HC:
                new_edge_list = extract_subgraph(edge_list, split_idx[i])
                GHG = Hypergraph(num_v=node_num, e_list=new_edge_list)
                # pre-training process
                if args.method == "FedHGN":
                    split_X[i] = GHG.smoothing_with_HGNN(split_X[i])
                    for _ in range(args.num_layers - 1):
                        split_X[i] = GHG.smoothing_with_HGNN(split_X[i])
            else:
                # pass
                new_edge_list, neighbors = extract_subgraph_with_neighbors(edge_list, split_idx[i])
                GHG = Hypergraph(num_v=node_num + len(neighbors), e_list=new_edge_list)
                split_point = split_X[i].shape[0]
                split_X[i] = features[split_idx[i] + neighbors]
                split_Y[i] = labels[split_idx[i] + neighbors]
                split_X[i] = GHG.smoothing_with_HGNN(split_X[i])    
                # split_X[i] = GHG.smoothing_with_HGNN(split_X[i])
                # split_X[i] = GHG.smoothing_with_HGNN(split_X[i])
                for _ in range(args.num_layers2 - 1):
                    split_X[i] = split_X[i][:split_point]
                    split_X[i] = torch.cat([split_X[i], new_features[neighbors]], dim=0)
                    split_X[i] = GHG.smoothing_with_HGNN(split_X[i])                               
                     
            split_structure.append(GHG)
        elif args.method == "HNHN":
            pass



    return split_X, split_Y, split_structure, split_train_mask, split_val_mask, split_test_mask

# Extract a subgraph containing only the client's own nodes
def extract_subgraph(edge_list, idx_list):
    # Create a mapping that maps the nodes in idx_list to new numbers
    # print(edge_list)
    old_to_new = {node: i for i, node in enumerate(idx_list)}
    new_edge_list = []  # The edge set of the new graph
    # Traverse each edge in the original graph
    for edge in edge_list:
        # If all nodes are in idx_list, add them to the new graph
        if all(node in old_to_new for node in edge):
            # Map old node numbers to new numbers
            new_edge_list.append(tuple(old_to_new[node] for node in edge))
        # If only some nodes are in idx_list, only some nodes are added (the remaining nodes are greater than or equal to 2)
        elif sum(node in old_to_new for node in edge) >= 2:
            new_edge_list.append(tuple(old_to_new[node] for node in edge if node in old_to_new))
    
    return new_edge_list

# Extract a subgraph containing the client's own nodes and the neighbor nodes of the client's border nodes
def extract_subgraph_with_neighbors(edge_list, idx_list, sub_edge_list=None):

    # Convert idx_list to a set so you can quickly check for elements
    idx_set = set(idx_list)
    included_nodes = set(idx_list)
    neighbors = set()

    # Traverse all edges to find all neighbors
    if sub_edge_list is None:
        for edge in edge_list:
            if any(node in idx_set for node in edge):
                neighbors.update(edge)
    else:
        for edge in sub_edge_list :
            # Cannot be cross-client edge (in multi-layer FedGCN)
            if any(node in idx_set for node in edge):
                neighbors.update(edge)
    
    # Exclude nodes that are already in idx_list and keep only real neighbor nodes
    neighbors.difference_update(idx_set)
    included_nodes.update(neighbors)
    
    # Create a new node mapping, with nodes in idx_list taking priority
    old_to_new = {node: i for i, node in enumerate(list(idx_list) + list(neighbors))}
    
    # Create a new edge set, using the new node numbers
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

# Randomly generate training, validation, and test masks
def rand_train_test_idx(node_num, train_ratio, val_ratio, test_ratio):

    trainCount = node_num * train_ratio
    valCount = node_num * val_ratio
    testCount = node_num * test_ratio
    train_mask = generate_bool_tensor(node_num, int(trainCount))
    val_mask = generate_bool_tensor(node_num, int(valCount), train_mask)
    test_mask = generate_bool_tensor(node_num, int(testCount), train_mask | val_mask)
    return train_mask, val_mask, test_mask

