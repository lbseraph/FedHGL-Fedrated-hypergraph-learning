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
from torch_geometric.nn import GCNConv, SAGEConv, SGConv
from dhg import Graph

class GCN(nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int, cached: bool
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
            self.convs.append(GCNConv(nfeat, nclass, normalize=True, add_self_loops=True, cached=True))
        else:      
            self.convs.append(GCNConv(nfeat, nhid, normalize=True, add_self_loops=True, cached=True))
            for _ in range(NumLayers - 2):
                self.convs.append(GCNConv(nhid, nhid, normalize=True, add_self_loops=True, cached=True))
            self.convs.append(GCNConv(nhid, nclass, normalize=True, add_self_loops=True, cached=True))
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
            self.hgcs.append(LineConv(in_ch, n_class))
        else:
            self.hgcs.append(LineConv(in_ch, n_hid))
            for _ in range(layer_num - 2):
                self.hgcs.append(LineConv(n_hid, n_hid))
            self.hgcs.append(LineConv(n_hid, n_class))
        self.act = nn.ReLU(inplace=True)
        self.layer_num = layer_num
        self.drop = nn.Dropout(dropout)
        
    def reset_parameters(self):
        for hgc in self.hgcs:
            hgc.reset_parameters()

    def forward(self, x, _G):
        for hgc in self.hgcs[:-1]:
            x = hgc(x)
            x = self.drop(x)
        x = self.hgcs[-1](x)
        x = self.act(x)

        # r = torch.log_softmax(x, dim=-1)
        return torch.log_softmax(x, dim=-1)

class SGC(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, layer_num=2, dropout=0.5):
        super(SGC, self).__init__()
        self.hgcs = torch.nn.ModuleList()
        if layer_num == 1:
            self.hgcs.append(LineConv(in_ch, n_class))
        else:
            self.hgcs.append(LineConv(in_ch, n_hid))
            for _ in range(layer_num - 2):
                self.hgcs.append(LineConv(n_hid, n_hid))
            self.hgcs.append(LineConv(n_hid, n_class))
        self.act = nn.ReLU(inplace=True)
        self.layer_num = layer_num
        self.drop = nn.Dropout(dropout)
        
    def reset_parameters(self):
        for hgc in self.hgcs:
            hgc.reset_parameters()

    def forward(self, x, _G):
        for hgc in self.hgcs[:-1]:
            x = hgc(x)
            x = self.drop(x)
        x = self.hgcs[-1](x)
        x = self.act(x)

        # r = torch.log_softmax(x, dim=-1)
        return torch.log_softmax(x, dim=-1)

class HNHN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, layer_num=2, dropout=0.5):
        super(HNHN, self).__init__()
        self.hgcs = torch.nn.ModuleList()
        if layer_num == 1:
            self.hgcs.append(HNHNConv(in_ch, n_class, drop_rate=dropout, is_last=True))
        else:
            self.hgcs.append(HNHNConv(in_ch, n_hid, drop_rate=dropout, is_last=False))
            for _ in range(layer_num - 2):
                self.hgcs.append(HNHNConv(n_hid, n_hid, drop_rate=dropout, is_last=False))
            self.hgcs.append(HNHNConv(n_hid, n_class, drop_rate=dropout, is_last=True))
        
    def reset_parameters(self):
        for hgc in self.hgcs:
            hgc.reset_parameters()

    def forward(self, x, G):
        for hgc in self.hgcs[:-1]:
            x = hgc(x, G)
        x = self.hgcs[-1](x, G)

        # r = torch.log_softmax(x, dim=-1)
        return torch.log_softmax(x, dim=-1)


class HNHNConv(nn.Module):
    r"""The HNHN convolution layer proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta_v2e = LineConv(in_channels, out_channels)
        self.theta_e2v = LineConv(out_channels, out_channels)

    def reset_parameters(self):
        self.theta_v2e.reset_parameters()
        self.theta_e2v.reset_parameters()

    def forward(self, X, hg) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        # v -> e
        X = self.theta_v2e(X)
        Y = self.act(hg.v2e(X, aggr="mean"))
        # e -> v
        Y = self.theta_e2v(Y)
        X = hg.e2v(Y, aggr="mean")
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X


class LineConv(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(LineConv, self).__init__()
        self.lin = nn.Linear(in_ft, out_ft, bias=False)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, X):
        # X = hg.smoothing_with_HGNN(X) # No need to use HGNN smoothing because of pre-training
        X = self.lin(X)
        return X

class HyperGCN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, layer_num=2, dropout=0.5):
        super(HyperGCN, self).__init__()
        self.hgcs = torch.nn.ModuleList()
        if layer_num == 1:
            self.hgcs.append(HyperGCNConv(in_ch, n_class, drop_rate=dropout, is_last=True))
        else:
            self.hgcs.append(HyperGCNConv(in_ch, n_hid, drop_rate=dropout, is_last=False))
            for _ in range(layer_num - 2):
                self.hgcs.append(HyperGCNConv(n_hid, n_hid, drop_rate=dropout, is_last=False))
            self.hgcs.append(HyperGCNConv(n_hid, n_class, drop_rate=dropout, is_last=True))
        
    def reset_parameters(self):
        for hgc in self.hgcs:
            hgc.reset_parameters()

    def forward(self, x, G):
        for hgc in self.hgcs[:-1]:
            x = hgc(x, G)
        x = self.hgcs[-1](x, G)

        # r = torch.log_softmax(x, dim=-1)
        return torch.log_softmax(x, dim=-1)

class HyperGCNConv(nn.Module):
    r"""The HyperGCN convolution layer proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mediator: bool = False,
        bias: bool = False,
        use_bn: bool = True,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.use_mediator = use_mediator
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.theta.reset_parameters()

    def forward(
        self, X: torch.Tensor, hg,
    ) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
            ``cached_g`` (``dhg.Graph``): The pre-transformed graph structure from the hypergraph structure that contains :math:`N` vertices. If not provided, the graph structure will be transformed for each forward time. Defaults to ``None``.
        """
        X = self.theta(X)

        g = Graph.from_hypergraph_hypergcn(
            hg, X, self.use_mediator, device=X.device
        )
        X = g.smoothing_with_GCN(X)

        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X

