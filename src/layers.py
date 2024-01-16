#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains layers used in AllSet and all other tested methods.
"""

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.lin = Linear(in_ft, out_ft, bias=bias)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, G):
        x = self.lin(x)
        x = torch.matmul(G, x)
        return x
