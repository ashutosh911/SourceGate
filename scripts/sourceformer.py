import torch
import torch.nn as nn
import torch.nn.functional as F

# K=3 taxonomy
SOURCE_TYPES = ["text", "table", "kg"]
SOURCE_TYPE_IDX = {s: i for i, s in enumerate(SOURCE_TYPES)}
K = len(SOURCE_TYPES)
EMBED_DIM = 768

DATASET_TO_TYPE = {
    "nq": "text", "triviaqa": "text",
    "ott": "table", "tat": "table",
    "kg": "kg",
}
TYPE_TO_DATASETS = {
    "text": ["nq", "triviaqa"],
    "table": ["ott", "tat"],
    "kg": ["kg"],
}

# Backward compat for K=5 baselines
SOURCES = ["nq", "triviaqa", "ott", "tat", "kg"]
SOURCE_TO_IDX = {s: i for i, s in enumerate(SOURCES)}


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """Gumbel-Softmax with straight-through estimator."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    y_soft = F.softmax((logits + gumbels) / tau, dim=dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class SourceFormerK3(nn.Module):
    """K=3 routing MLP: query embedding → {text, table, kg} logits."""
    def __init__(self, input_dim=EMBED_DIM, hidden=512, mid=128, k=K, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, mid), nn.LayerNorm(mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, k),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x).argmax(-1)

    def route(self, x, tau=1.0, hard=True):
        return gumbel_softmax(self.forward(x), tau=tau, hard=hard)

# Add to sourceformer.py after SourceFormerK3

# K=5 taxonomy
K5 = 5
SOURCE_TYPES_K5 = ["nq", "triviaqa", "ott", "tat", "kg"]
SOURCE_TYPE_IDX_K5 = {s: i for i, s in enumerate(SOURCE_TYPES_K5)}

# Each K=5 type maps directly to one dataset (no merging)
TYPE_TO_DATASET_K5 = {
    "nq": "nq", "triviaqa": "triviaqa",
    "ott": "ott", "tat": "tat", "kg": "kg",
}


class SourceFormerK5(nn.Module):
    """K=5 routing MLP: query embedding → {nq, triviaqa, ott, tat, kg} logits."""
    def __init__(self, input_dim=EMBED_DIM, hidden=512, mid=128, k=K5, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, mid), nn.LayerNorm(mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, k),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x).argmax(-1)

    def route(self, x, tau=1.0, hard=True):
        return gumbel_softmax(self.forward(x), tau=tau, hard=hard)

# Legacy K=5 variants kept for ablations
class MLPSourceFormer(nn.Module):
    def __init__(self, input_dim=EMBED_DIM, hidden=512, num_sources=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden // 2, num_sources)
        )

    def forward(self, x):
        return self.net(x)


def build_sourceformer(variant="k3", k=K):
    if variant == "k3":
        return SourceFormerK3(k=k)
    if variant == "mlp":
        return MLPSourceFormer(num_sources=k)
    raise ValueError(f"Unknown variant: {variant}")
