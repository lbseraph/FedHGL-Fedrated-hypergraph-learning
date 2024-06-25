#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if NumLayers == 1:
            self.convs.append(GCNConv(nfeat, nclass, normalize=True, cached=True))
        else:      
            self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=True))
            for _ in range(NumLayers - 2):
                self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
            self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=True))
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t):
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
            x = self.act(x)
            x = self.drop(x)
            
        x = self.convs[-1](x, adj_t)
        
        return x

class SAGE(nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network
        """
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
        self.convs.append(SAGEConv(nhid, nclass))

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

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, layer_num=2, dropout=0.5):
        super(HGNN, self).__init__()
        self.hgcs = torch.nn.ModuleList()
        if layer_num == 1:
            self.hgcs.append(HGNN_conv(in_ch, n_class))
        else:
            self.hgcs.append(HGNN_conv(in_ch, n_hid))
            for _ in range(layer_num - 2):
                self.hgcs.append(HGNN_conv(n_hid, n_hid))
            self.hgcs.append(HGNN_conv(n_hid, n_class))
        self.act = nn.ReLU(inplace=True)
        self.layer_num = layer_num
        self.dropout = dropout
        
    def reset_parameters(self):
        for hgc in self.hgcs:
            hgc.reset_parameters()

    def forward(self, x, G):
        for hgc in self.hgcs:
            x = hgc(x, G)
        x = self.act(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # r = torch.log_softmax(x, dim=-1)
        return x

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(HGNN_conv, self).__init__()
        self.lin = nn.Linear(in_ft, out_ft, bias=False)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, X, _hg):
        # X = hg.smoothing_with_HGNN(X) # No need to use HGNN smoothing because of pre-training
        X = self.lin(X)
        return X