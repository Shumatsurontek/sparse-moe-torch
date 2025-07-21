import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sparse_moe.hf_model import MiniMoEConfig, MiniMoEHFModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SparseMoE")

# Sparse MoE layer
class SparseMoE(nn.Module):
    """
    Sparse Mixture-of-Experts (MoE) layer.

    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Hidden dimension for each expert MLP.
        num_experts (int): Number of expert networks.
        top_k (int): Number of experts to route each token to.

    Forward:
        x (Tensor): Shape (batch_size, seq_len, input_dim)
        Returns: Tensor of same shape as input.

    Routing:
        Each token is routed to top_k experts based on router logits.
        Only top_k experts process each token, weighted by softmax over their logits.
    """
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Define the experts (feedforward networks)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])

        # Router: just a linear projection for now
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, dim = x.size()
        num_tokens = batch_size * seq_len
        logger.info(f"Input shape: {x.shape}, num_tokens: {num_tokens}")

        # Flatten inputs to (batch * seq, dim)
        x_flat = x.view(-1, dim)
        routing_logits = self.router(x_flat)  # (batch * seq, num_experts)
        logger.info(f"Routing logits shape: {routing_logits.shape}")

        # Select top-k experts per token
        top_k_vals, top_k_indices = torch.topk(routing_logits, self.top_k, dim=-1)  # (batch * seq, k)
        logger.info(f"Top-k indices: {top_k_indices}")
        logger.info(f"Top-k values: {top_k_vals}")

        # Normalize logits with softmax (only over top-k)
        top_k_gates = F.softmax(top_k_vals, dim=-1)  # (batch * seq, k)
        logger.info(f"Top-k gates (softmax over top-k): {top_k_gates}")

        # Initialize output
        output = torch.zeros_like(x_flat)
        token_counts = torch.zeros(self.num_experts, dtype=torch.int32)

        # Dispatch to each selected expert
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # (batch * seq)
            mask = F.one_hot(expert_idx, self.num_experts).float()  # (batch * seq, num_experts)

            for expert_id in range(self.num_experts):
                # Find tokens routed to this expert
                tokens_for_expert = mask[:, expert_id].nonzero(as_tuple=False).squeeze(1)
                n_tokens = tokens_for_expert.numel()
                if n_tokens == 0:
                    continue
                token_counts[expert_id] += n_tokens
                expert_input = x_flat[tokens_for_expert]
                expert_output = self.experts[expert_id](expert_input)

                gate_values = top_k_gates[tokens_for_expert, k].unsqueeze(1)
                output[tokens_for_expert] += gate_values * expert_output
                logger.info(f"Expert {expert_id}: tokens processed at k={k}: {n_tokens}")
        logger.info(f"Token count per expert: {token_counts}")
        return output.view(batch_size, seq_len, dim)


# Transformer block with MoE
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, dim, heads, ff_hidden_dim, num_experts=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = SparseMoE(dim, ff_hidden_dim, num_experts=num_experts)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        moe_output = self.moe(x)
        return self.norm2(x + moe_output)

# Mini MoE transformer
class MiniMoETransformer(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, ff_hidden=256, num_experts=4):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockWithMoE(dim, heads, ff_hidden, num_experts)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Test/demo
batch_size = 16
seq_len = 1024
dim = 512
num_experts = 16
top_k = 4

model = MiniMoETransformer(dim=dim, num_experts=num_experts)
dummy_input = torch.randn(batch_size, seq_len, dim)

output = model(dummy_input)
print("Output shape:", output.shape)

# Pour observer la distribution des tokens par expert, on refait le routage ici
moe = model.layers[0].moe
with torch.no_grad():
    x = dummy_input
    x_flat = x.view(-1, dim)
    routing_logits = moe.router(x_flat)
    top_k_vals, top_k_indices = torch.topk(routing_logits, top_k, dim=-1)
    counts = torch.zeros(num_experts, dtype=torch.int32)
    for k in range(top_k):
        expert_ids = top_k_indices[:, k]
        for i in range(num_experts):
            counts[i] += (expert_ids == i).sum()
    logger.info(f"[Test] Token count per expert: {counts}")


