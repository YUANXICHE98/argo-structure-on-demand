import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x):
        return self.net(x)


class ConditionedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.h0 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

    def forward(self, forcing, static_state):
        _, h = self.gru(forcing, self.h0(static_state).unsqueeze(0))
        return h.squeeze(0)


class HypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.node = nn.Linear(in_dim, out_dim)
        self.edge = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, incidence):
        h = incidence.clamp(min=0)
        dv = h.sum(1, keepdim=True).clamp(min=1e-8)
        de = h.sum(0, keepdim=True).t().clamp(min=1e-8)
        edge_state = torch.mm(h.t(), x) / de
        out = torch.mm(h, self.edge(edge_state)) / dv
        return self.norm(F.gelu(self.node(x) + self.dropout(out)))
