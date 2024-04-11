#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

import torch.nn as nn
from layers import *
from torch_geometric.nn import GCNConv, SAGEConv

class GCN(nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the GCN model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.
        """
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor):
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)

# class HGNN(nn.Module):
#     def __init__(self, in_ch, n_class, n_hid, dropout=0.5, layer_num=2):
#         super(HGNN, self).__init__()
#         self.dropout = dropout
#         self.hgc1 = HGNN_conv(in_ch, n_hid)
#         self.hgc2 = HGNN_conv(n_hid, n_class)

#     def reset_parameters(self):
#         self.hgc1.reset_parameters()
#         self.hgc2.reset_parameters()

#     def forward(self, x, G):

#         x = F.relu(self.hgc1(x, G))
#         x = F.dropout(x, self.dropout)
#         x = self.hgc2(x, G)
#         return x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5, layer_num=2):
        super(HGNN, self).__init__()
        self.hgc = HGNN_conv(in_ch, n_class)
        self.layer_num = layer_num
        self.n_class = n_class
        self.in_ch = in_ch
    def reset_parameters(self):
        self.hgc.reset_parameters()
#         self.hgc2.reset_parameters()

    def forward(self, x, G):
        for _ in range(self.layer_num - 1):
            x = G.matmul(x)
        x = self.hgc(x, G)
        # print(x.shape, data.x.shape, self.n_class, self.in_ch)
        return x
