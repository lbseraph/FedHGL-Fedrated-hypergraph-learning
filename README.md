# code for "Federated Hypergraph Learning with Hyperedge Completion"

## Description:

This repository contains code to reproduce "Federated Hypergraph Learning: Hyperedge Completion with Local
Differential Privacy". This paper introduces a novel federated hypergraph learning method: FedHGN. It can effectively address the challenges of distributed training of high-order graph data. Experimental results on real-world datasets confirm FedHGN’s effectiveness and its competitive edge over existing federated subgraph learning methods.

## Datasets

* hypergraph datasets: Cora-CA, DBLP4k, IMDB4k, 20News
* simple graph: Cora, CiteSeer, Facebook

## How to Run

1.Install Python 3.9 and the necessary dependencies with the command ``pip install -r requirements.txt``.

2.Select from seven datasets(Cora-CA, etc.) and set the number of clients.

3.Run the code. (See detailed examples below). On hypergraph datasets: e.g. ``python ./src/main.py --dname cora-ca --num_layers 2 --n_client 1 --method FedHGN --global_rounds 600`` Use 1 client to perform 600 rounds on the cora-ca dataset to reproduce the experimental results. On simple graph datasets: e.g. ``python ./src/main.py --dname cora --num_layers 2 --n_client 3 --method FedGCN`` Use 3 clients to reproduce the experimental results using the FedGCN algorithm on the cora dataset.

