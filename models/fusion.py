import torch
import torch.nn as nn

class MultiHeadFusion(nn.Module):
    def __init__(self, env_dim, snp_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.q = nn.Linear(env_dim, hidden_dim)
        self.k = nn.Linear(snp_dim, hidden_dim)
        self.v = nn.Linear(snp_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim + env_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
    def forward(self, env_feat, snp_feat):
        Q = self.q(env_feat).unsqueeze(1)
        K = self.k(snp_feat).unsqueeze(1)
        V = self.v(snp_feat).unsqueeze(1)
        attn_out, _ = self.attn(Q, K, V)
        fusion = torch.cat([env_feat, attn_out.squeeze(1)], dim=1)
        return self.out(fusion).squeeze(1)

class MetaTraitFusion(nn.Module):
    def __init__(self, num_traits):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(num_traits, 64),
            nn.ReLU(),
            nn.Linear(64, num_traits),
            nn.Softmax(dim=1)
        )
        self.out = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, trait_outputs):
        weights = self.attn(trait_outputs.view(trait_outputs.size(0), -1))
        weighted = torch.sum(weights.unsqueeze(-1) * trait_outputs, dim=1)
        return self.out(weighted).squeeze(1), weights