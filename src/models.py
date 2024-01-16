#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

import torch.nn as nn
from layers import *

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
