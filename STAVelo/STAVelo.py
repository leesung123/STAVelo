import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .gat_conv import GATConv


class STAVelo(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STAVelo, self).__init__()

        [feature_dim, num_hid_1, num_hid_2, num_hid_3] = hidden_dims
        self.feature_dim = feature_dim
        self.conv1 = GATConv(feature_dim*2, num_hid_1, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hid_1, num_hid_2, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(num_hid_2, num_hid_3, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hid_3, feature_dim*3, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        raw_out = self.conv4(h3, edge_index, attention=False)

        fd = self.feature_dim
        alpha_raw = raw_out[:, 0:fd]
        beta_raw = raw_out[:, fd:fd*2]
        gamma_raw = raw_out[:, fd*2:fd*3]

        alpha_out = F.softplus(alpha_raw) + 1e-6
        beta_out = F.softplus(beta_raw) + 1e-6
        gamma_out = F.softplus(gamma_raw) + 1e-6

        out = torch.cat([alpha_out, beta_out, gamma_out], dim=-1)
        return h1, h2, h3, out