import torch
import torch.nn as nn
import torch.nn.functional as F

class HGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_node = nn.Parameter(torch.empty(out_dim, 1))
        self.attn_edge = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_uniform_(self.attn_node)
        nn.init.xavier_uniform_(self.attn_edge)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, X, H):
        X = self.linear(X)
        Z = torch.matmul(H.T, X)
        node_score = torch.matmul(X, self.attn_node)
        edge_score = torch.matmul(Z, self.attn_edge)
        attn_ve = self.leaky_relu(torch.matmul(H, edge_score) + node_score)
        attn_ve = F.softmax(attn_ve * H, dim=1)
        attn_ve = self.dropout(attn_ve)
        out = torch.matmul(attn_ve, Z)
        return self.layer_norm(out + X)

class EnvHGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim_1=2048, hidden_dim=128, out_dim=64):
        super().__init__()
        self.layer1 = HGATLayer(in_dim, hidden_dim_1)
        self.layer2 = HGATLayer(hidden_dim_1, hidden_dim)
        self.layer3 = HGATLayer(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, X, H):
        X = self.layer1(X, H)
        X = self.layer2(X, H)
        X = self.layer3(X, H)
        return self.out_proj(X)