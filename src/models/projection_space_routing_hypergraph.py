import torch
import torch.nn as nn
import torch.nn.functional as F
from .hypergraph_hydro import ConditionedGRU, HypergraphConv, StaticEncoder
from .sparse_structure_moe_hypergraph import ViewConditionedSparseHypergraphGenerator


class ProjectionSpaceRouter(nn.Module):
    def __init__(self, hidden_dim, num_spaces, top_k=2):
        super().__init__()
        self.num_spaces = num_spaces
        self.top_k = min(top_k, num_spaces)
        self.space_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_spaces))
        self.deviation_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, static_mean, temporal_mean, disagreement):
        x = torch.cat([static_mean, temporal_mean, disagreement], dim=-1)
        logits = self.space_gate(x)
        if self.top_k < self.num_spaces:
            vals, idx = torch.topk(logits, self.top_k, dim=-1)
            sparse = torch.full_like(logits, -1e9)
            sparse.scatter_(0, idx, vals)
            probs = F.softmax(sparse, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
        beta = torch.sigmoid(self.deviation_gate(x)).squeeze(-1)
        return probs, beta


class ProjectionSpaceRoutingHyperNet(nn.Module):
    def __init__(self, num_basins, num_forcing_features, num_static_features, hidden_dim=64, num_adapt_edges=32, num_hgconv_layers=2, num_projection_spaces=3, top_k=2, dropout=0.1, edge_size_ratio=0.1, min_edge_size=8, max_edge_size=64, use_orthogonal_init=False, nce_loss_weight=0.0):
        super().__init__()
        if num_projection_spaces != 3:
            raise ValueError('This release model expects exactly three projection spaces.')
        self.num_projection_spaces = num_projection_spaces
        self.nce_loss_weight = nce_loss_weight
        self.static_encoder = StaticEncoder(num_static_features, hidden_dim)
        self.temporal_encoder = ConditionedGRU(num_forcing_features, hidden_dim)
        self.joint_view_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.delta_view_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        gen_kwargs = dict(hidden_dim=hidden_dim, num_hyperedges=num_adapt_edges, edge_size_ratio=edge_size_ratio, min_edge_size=min_edge_size, max_edge_size=max_edge_size, use_orthogonal_init=use_orthogonal_init)
        self.shared_generator = ViewConditionedSparseHypergraphGenerator(**gen_kwargs)
        self.projection_generators = nn.ModuleList([ViewConditionedSparseHypergraphGenerator(**gen_kwargs) for _ in range(num_projection_spaces)])
        self.router = ProjectionSpaceRouter(hidden_dim, num_projection_spaces, top_k=top_k)
        self.hg_convs = nn.ModuleList([HypergraphConv(hidden_dim, hidden_dim, dropout) for _ in range(num_hgconv_layers)])
        self.predictor = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _build_views(self, h_t, h_s):
        joint = self.joint_view_proj(torch.cat([h_t, h_s], dim=-1))
        delta = self.delta_view_proj(torch.cat([torch.abs(h_t - h_s), h_t * h_s], dim=-1))
        return h_s, [h_t, joint, delta]

    def _mix_projection_spaces(self, shared_hypergraph, space_hypergraphs, space_probs, beta):
        routed = (space_hypergraphs * space_probs.view(-1, 1, 1)).sum(dim=0)
        return (1.0 - beta) * shared_hypergraph + beta * routed

    def _structure_diversity_loss(self, hypergraphs):
        flat = F.normalize(hypergraphs.reshape(hypergraphs.shape[0], -1), dim=-1)
        sim = torch.mm(flat, flat.t())
        mask = 1 - torch.eye(sim.shape[0], device=sim.device)
        return (sim * mask).sum() / mask.sum().clamp(min=1.0)

    def _nce_space_loss(self, hypergraphs, temperature=0.1):
        flat = F.normalize(hypergraphs.reshape(hypergraphs.shape[0], -1), dim=-1)
        sim = torch.mm(flat, flat.t())
        pos = torch.diag(sim)
        neg = (sim.sum(1) - pos) / max(hypergraphs.shape[0] - 1, 1)
        return -torch.log(pos / (pos + temperature * neg + 1e-8)).mean()

    def _forward_projection_prediction(self, forcing, static_attrs):
        bsz, n, steps, fdim = forcing.shape
        h_s = self.static_encoder(static_attrs)
        h_t = self.temporal_encoder(forcing.reshape(bsz * n, steps, fdim), h_s.reshape(bsz * n, -1)).reshape(bsz, n, -1)
        preds = []
        losses = {k: 0.0 for k in ['graph_diversity_loss', 'graph_coverage_loss', 'structure_diversity_loss', 'beta_mean', 'nce_loss']}
        for b in range(bsz):
            static_view, views = self._build_views(h_t[b], h_s[b])
            shared, shared_aux = self.shared_generator(static_view)
            spaces, auxes = [], []
            for gen, view in zip(self.projection_generators, views):
                h, aux = gen(view)
                spaces.append(h); auxes.append(aux)
            spaces = torch.stack(spaces, dim=0)
            static_mean = static_view.mean(0)
            temporal_mean = views[0].mean(0)
            disagreement = torch.stack([torch.abs(v.mean(0) - static_mean) for v in views]).mean(0)
            probs, beta = self.router(static_mean, temporal_mean, disagreement)
            inc = self._mix_projection_spaces(shared, spaces, probs, beta)
            z = h_t[b]
            for conv in self.hg_convs:
                z = conv(z, inc)
            preds.append(self.predictor(torch.cat([z, h_s[b]], dim=-1)).squeeze(-1))
            losses['graph_diversity_loss'] = losses['graph_diversity_loss'] + shared_aux['graph_diversity_loss'] + torch.stack([a['graph_diversity_loss'] for a in auxes]).mean()
            losses['graph_coverage_loss'] = losses['graph_coverage_loss'] + shared_aux['graph_coverage_loss'] + torch.stack([a['graph_coverage_loss'] for a in auxes]).mean()
            losses['structure_diversity_loss'] = losses['structure_diversity_loss'] + self._structure_diversity_loss(spaces)
            losses['beta_mean'] = losses['beta_mean'] + beta
            if self.nce_loss_weight > 0:
                losses['nce_loss'] = losses['nce_loss'] + self._nce_space_loss(spaces)
        return torch.stack(preds, dim=0), {k: v / bsz for k, v in losses.items()}

    def forward(self, forcing, static_attrs, mask=None):
        return self._forward_projection_prediction(forcing, static_attrs)
