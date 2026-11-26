"""Incidence attention, genotype encoding and prediction-space fusion."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics.pairwise import euclidean_distances

def nearest(query, reference, k):
    if not len(reference) or k < 1:
        raise ValueError("The reference and neighbor count must be nonempty")
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("Reference topology requires finite transformed features")
    return np.concatenate([
        np.argsort(euclidean_distances(query[i:i+128], reference), axis=1, kind="stable")[:, :k]
        for i in range(0, len(query), 128)
    ])

def incidence_to_propagation(h, weights=None):
    """Compute Dv^-1/2 H W De^-1 H.T Dv^-1/2 for diagnostic comparisons."""
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("H must be a finite nonnegative node-by-edge matrix")
    weights = np.ones(h.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (h.shape[1],) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid hyperedge weights")
    node_degree, edge_degree = h @ weights, h.sum(0)
    inv_node = np.divide(1, np.sqrt(node_degree), out=np.zeros_like(node_degree), where=node_degree > 0)
    inv_edge = np.divide(weights, edge_degree, out=np.zeros_like(edge_degree), where=edge_degree > 0)
    return (inv_node[:, None] * h * inv_edge) @ (h.T * inv_node)

def masked_softmax(logits, mask):
    weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1)
    return weights * mask

@dataclass
class Reference:
    values: torch.Tensor
    node_for_edge: torch.Tensor
    edge_for_node: torch.Tensor

    @classmethod
    def build(cls, ref, k, device):
        # One hyperedge per training identifier with guaranteed self-incidence.
        k = min(k, len(ref))
        if k < 1:
            raise ValueError("Empty reference graph or invalid neighbor count")
        proposals = nearest(ref, ref, min(k + 1, len(ref)))
        nodes = np.array([[i] + [int(j) for j in row if j != i][:k-1]
                          for i, row in enumerate(proposals)])
        reverse = [[] for _ in ref]
        for edge, row in enumerate(nodes):
            for node in row:
                reverse[node].append(edge)
        padding = np.full((len(ref), max(map(len, reverse))), -1, dtype=np.int64)
        for i, edges in enumerate(reverse):
            padding[i, :len(edges)] = edges
        return cls(torch.as_tensor(ref, dtype=torch.float32, device=device),
                   torch.as_tensor(nodes, dtype=torch.long, device=device), torch.as_tensor(padding, dtype=torch.long, device=device))

@dataclass
class Query:
    values: torch.Tensor
    seen: torch.Tensor
    neighbors: torch.Tensor

    def __getitem__(self, index):
        return Query(self.values[index], self.seen[index], self.neighbors[index])

class IncidenceAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.node_key = nn.Linear(width, width, bias=False)
        self.center_query = nn.Linear(width, width, bias=False)
        self.edge_key = nn.Linear(width, width, bias=False)
        self.node_query = nn.Linear(width, width, bias=False)
        self.edge_value = nn.Linear(width, width, bias=False)
        self.message = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.scale = width ** -.5

    def aggregate(self, query, values, incidence, query_layer, key_layer):
        valid = incidence >= 0
        indexed = values[incidence.clamp_min(0)]
        logits = (query_layer(query)[:, None, :] * key_layer(indexed)).sum(-1) * self.scale
        weights = masked_softmax(logits, valid)
        return (weights[..., None] * indexed).sum(1), weights

    def forward(self, nodes, topology, query, query_neighbors):
        edge, _ = self.aggregate(nodes, nodes, topology.node_for_edge, self.center_query, self.node_key)
        ref_values = self.edge_value(edge)
        reference_message, _ = self.aggregate(nodes, ref_values, topology.edge_for_node,
                                              self.node_query, self.edge_key)
        query_message, _ = self.aggregate(query, ref_values, query_neighbors,
                                          self.node_query, self.edge_key)
        return (self.norm(nodes + self.message(reference_message)),
                self.norm(query + self.message(query_message)))

class EnvHGAT(nn.Module):
    def __init__(self, inputs, width, layers=2):
        super().__init__()
        self.project = nn.Linear(inputs, width)
        self.layers = nn.ModuleList(IncidenceAttention(width) for _ in range(layers))

    def forward(self, reference, query):
        nodes = F.gelu(self.project(reference.values))
        targets = F.gelu(self.project(query.values))
        for layer in self.layers:
            nodes, targets = layer(nodes, reference, targets, query.neighbors)
            targets = torch.where(query.seen[:, None] >= 0, nodes[query.seen.clamp_min(0)], targets)
        return targets

class SNPEncoderCNN(nn.Module):
    def __init__(self, width, tokens):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, width, 7, stride=2, padding=3), nn.GELU(),
                                  nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
                                  nn.AdaptiveAvgPool1d(tokens))

    def forward(self, values):
        return self.conv(values[:, None, :]).transpose(1, 2)

class TokenCrossAttentionFusion(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by the number of heads")
        self.heads = heads
        self.dimension = width // heads
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.project = nn.Linear(width, width)
        self.output = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout))

    def forward(self, env, tokens, mask=None, return_weights=False):
        if mask is not None and mask.all(dim=1).any():
            raise ValueError("Each sample requires a visible SNP token")
        batch, length, width = tokens.shape
        query = self.query(env).view(batch, self.heads, 1, self.dimension)
        key = self.key(tokens).view(batch, length, self.heads, self.dimension).transpose(1, 2)
        value = self.value(tokens).view(batch, length, self.heads, self.dimension).transpose(1, 2)
        logits = (query @ key.transpose(-2, -1)) * self.dimension ** -.5
        weights = logits.softmax(-1) if mask is None else masked_softmax(logits, ~mask[:, None, None])
        context = self.project((weights @ value).reshape(batch, width))
        output = self.output(torch.cat((env, context), 1))
        return (output, weights) if return_weights else output

class LegacyMultiHeadFusion(nn.Module):
    """Single-key diagnostic matching the historical fusion calculation."""
    def __init__(self, width, heads):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(width, width) for _ in range(3))
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, env, tokens, return_weights=False):
        feature = tokens.mean(1)
        context, weights = self.attention(self.q(env)[:, None], self.k(feature)[:, None],
                                          self.v(feature)[:, None], need_weights=True)
        return (context[:, 0], weights) if return_weights else context[:, 0]

class ConcatFusion(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.output = nn.Sequential(nn.Linear(width * 2, width * 2), nn.GELU(),
                                    nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout))

    def forward(self, env, tokens):
        return self.output(torch.cat((env, tokens.mean(1)), 1))

class BaseModel(nn.Module):
    def __init__(self, env_size, snp_size, traits, params, variant):
        super().__init__()
        width, tokens, dropout = (params[k] for k in ("width", "tokens", "dropout"))
        self.env_graph = variant != "wo_HGAT"
        self.snp_graph = variant == "wo_CNN"
        self.env = EnvHGAT(env_size, width, params["graph_layers"]) if self.env_graph else nn.Sequential(
            nn.Linear(env_size, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.snp = EnvHGAT(snp_size, width, params["graph_layers"]) if self.snp_graph else SNPEncoderCNN(width, tokens)
        self.snp_tokens = nn.ModuleList(nn.Linear(width, width) for _ in range(tokens)) if self.snp_graph else None
        if variant == "wo_CrossAttention":
            self.fusion = ConcatFusion(width, dropout)
        elif variant == "legacy_fusion":
            self.fusion = LegacyMultiHeadFusion(width, params["heads"])
        else:
            self.fusion = TokenCrossAttentionFusion(width, params["heads"], dropout)
        self.heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(traits))

    def forward(self, env, snp, env_ref, snp_ref):
        env_encoded = self.env(env_ref, env) if self.env_graph else self.env(env.values)
        if self.snp_graph:
            encoded = self.snp(snp_ref, snp)
            tokens = torch.stack([proj(encoded) for proj in self.snp_tokens], 1)
        else:
            tokens = self.snp(snp.values)
        hidden = self.fusion(env_encoded, tokens)
        return torch.cat([head(hidden) for head in self.heads], 1)

class MetaTraitFusion(nn.Module):
    def __init__(self, traits, width, dropout):
        super().__init__()
        self.traits = traits
        self.attention = nn.Sequential(nn.Linear(traits, width), nn.GELU(),
                                       nn.Linear(width, traits * traits))
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(traits, width), nn.GELU(),
                                                 nn.Dropout(dropout), nn.Linear(width, 1))
                                   for _ in range(traits))

    def forward(self, base, return_weights=False):
        if base.ndim != 3 or base.shape[1:] != (self.traits, 1):
            raise ValueError("Expected [samples, traits, 1]")
        weights = self.attention(base[..., 0]).view(-1, self.traits, self.traits).softmax(-1)
        weighted = base.transpose(1, 2) * weights
        prediction = torch.cat([head(weighted[:, i, :]) for i, head in enumerate(self.heads)], 1)
        return (prediction, weights) if return_weights else prediction
