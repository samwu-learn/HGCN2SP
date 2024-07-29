import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn.functional import cosine_similarity
import scipy.sparse as sp
import numpy as np
from torch_geometric.data import Data
import math
import os

class Encoder(nn.Module):
    def __init__(self, var_dim, con_dim, low_hid_dim, scenario_dim, high_hid_dim, high_out_dim, activation='prelu'):
        super(Encoder, self).__init__()
        self.mlp_var = nn.Linear(var_dim, low_hid_dim)
        self.mlp_con = nn.Linear(con_dim, low_hid_dim)
        self.conv1 = GCNConv(low_hid_dim, scenario_dim)
        self.conv2 = GCNConv(scenario_dim, high_hid_dim)
        self.conv3 = GCNConv(high_hid_dim, high_out_dim)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(low_hid_dim)
        self.bn2 = nn.BatchNorm1d(scenario_dim)
        self.bn3 = nn.BatchNorm1d(high_hid_dim)
        self.init_weights()

    def init_weights(self):
        # 使用正交初始化对线性层进行初始化
        nn.init.orthogonal_(self.mlp_var.weight)
        nn.init.orthogonal_(self.mlp_con.weight)
        if self.mlp_var.bias is not None:
            nn.init.zeros_(self.mlp_var.bias)
        if self.mlp_con.bias is not None:
            nn.init.zeros_(self.mlp_con.bias)
    
        # 对GCN层进行初始化
        # 注意：GCNConv的权重初始化可能需要根据实现细节进行调整
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.orthogonal_(conv.lin.weight)


    def filter_adj(self, adj, thre):
        adj[adj < thre] = 0.
        adj = adj.cpu().numpy()
        scenario_adj = sp.coo_matrix(adj)
        scenario_attr = torch.from_numpy(scenario_adj.data).cuda()
        scenario_adj = torch.stack([torch.from_numpy(scenario_adj.row), torch.from_numpy(scenario_adj.col)], dim=0).cuda()
        
        return scenario_adj, scenario_attr
        
    def forward(self, graph, thre=0.7):#x, edge_index, batch,  edge_adj, edge_adj_attr, edge_attr):
        
        mask_var = (graph.mark==0)
        mask_con = (graph.mark==1)
        
        var_feat = self.act(self.mlp_var(graph.x[mask_var]))
        con_feat = self.act(self.mlp_con(graph.x[mask_con][:,0:1]))
        
        x = torch.cat((con_feat, var_feat), dim=0)
        x = self.bn1(x)
        
        x = self.dropout(x)  

        x = self.conv1(x, graph.edge_index, graph.edge_attr).float()

        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # 全局平均池化  1 200 dim
        x = global_mean_pool(x, graph.batch).float()

        edge_scenario,attr_scenario = self.filter_adj(graph.scen_adj, thre)
        
        g = Data(x=x, edge_index= edge_scenario, edge_attr= attr_scenario)

        feat = self.conv2(g.x, g.edge_index).float()

        feat = self.bn3(feat)
        
        feat = self.act(feat)

        feat =self.dropout(feat)
        
        feat = self.act(self.conv3(feat, g.edge_index)).float()
        
        return feat, torch.mean(feat, 0) 


