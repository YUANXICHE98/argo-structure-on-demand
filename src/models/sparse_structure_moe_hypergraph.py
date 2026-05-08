import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewConditionedSparseHypergraphGenerator(nn.Module):
    def __init__(self, hidden_dim, num_hyperedges=32, edge_size_ratio=0.1, min_edge_size=8, max_edge_size=64, use_orthogonal_init=False):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.edge_size_ratio = edge_size_ratio
        self.min_edge_size = min_edge_size
        self.max_edge_size = max_edge_size
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        queries = torch.randn(num_hyperedges, hidden_dim) * 0.02
        if use_orthogonal_init:
            with torch.no_grad():
                if num_hyperedges < hidden_dim:
                    padded = torch.zeros(hidden_dim, hidden_dim)
                    padded[:num_hyperedges] = queries
                    q, _ = torch.linalg.qr(padded)
                    queries = q[:num_hyperedges]
                else:
                    queries, _ = torch.linalg.qr(queries)
        self.edge_queries = nn.Parameter(queries)

    def forward(self, node_state):
        n = node_state.shape[0]
        k = min(n, max(self.min_edge_size, min(self.max_edge_size, int(n * self.edge_size_ratio))))
        node = F.normalize(self.node_proj(node_state), dim=-1)
        query = F.normalize(self.edge_queries, dim=-1)
        scores = torch.mm(node, query.t())
        vals, idx = torch.topk(scores, k=k, dim=0)
        weights = F.softmax(vals, dim=0)
        h = torch.zeros_like(scores)
        h.scatter_(0, idx, weights)
        query_sim = torch.mm(query, query.t())
        eye = torch.eye(query_sim.shape[0], device=query_sim.device)
        diversity = ((query_sim - eye) ** 2 * (1 - eye)).sum() / (1 - eye).sum().clamp(min=1.0)
        coverage = F.relu(1.0 - (h > 0).float().sum(1)).mean()
        return h, {"graph_diversity_loss": diversity, "graph_coverage_loss": coverage, "edge_size": float(k)}
