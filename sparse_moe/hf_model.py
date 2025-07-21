import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import PreTrainedModel, PretrainedConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SparseMoE")

class MiniMoEConfig(PretrainedConfig):
    model_type = "mini-moe"
    def __init__(
        self,
        dim=128,
        depth=4,
        heads=4,
        ff_hidden=256,
        num_experts=4,
        top_k=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.ff_hidden = ff_hidden
        self.num_experts = num_experts
        self.top_k = top_k

class SparseMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_tokens = batch_size * seq_len
        #logger.info(f"Input shape: {x.shape}, num_tokens: {num_tokens}")

        x_flat = x.view(-1, dim)
        routing_logits = self.router(x_flat)
        #logger.info(f"Routing logits shape: {routing_logits.shape}")

        top_k_vals, top_k_indices = torch.topk(routing_logits, self.top_k, dim=-1)
        #logger.info(f"Top-k indices: {top_k_indices}")
        #logger.info(f"Top-k values: {top_k_vals}")

        top_k_gates = F.softmax(top_k_vals, dim=-1)
        #logger.info(f"Top-k gates (softmax over top-k): {top_k_gates}")

        output = torch.zeros_like(x_flat)
        token_counts = torch.zeros(self.num_experts, dtype=torch.int32)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            mask = F.one_hot(expert_idx, self.num_experts).float()
            for expert_id in range(self.num_experts):
                tokens_for_expert = mask[:, expert_id].nonzero(as_tuple=False).squeeze(1)
                n_tokens = tokens_for_expert.numel()
                if n_tokens == 0:
                    continue
                token_counts[expert_id] += n_tokens
                expert_input = x_flat[tokens_for_expert]
                expert_output = self.experts[expert_id](expert_input)
                gate_values = top_k_gates[tokens_for_expert, k].unsqueeze(1)
                output[tokens_for_expert] += gate_values * expert_output
                #logger.info(f"Expert {expert_id}: tokens processed at k={k}: {n_tokens}")
        #logger.info(f"Token count per expert: {token_counts}")
        return output.view(batch_size, seq_len, dim)

class TransformerBlockWithMoE(nn.Module):
    def __init__(self, dim, heads, ff_hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = SparseMoE(dim, ff_hidden_dim, num_experts=num_experts, top_k=top_k)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        moe_output = self.moe(x)
        return self.norm2(x + moe_output)

class MiniMoETransformer(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, ff_hidden=256, num_experts=4, top_k=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockWithMoE(dim, heads, ff_hidden, num_experts, top_k)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MiniMoEHFModel(PreTrainedModel):
    config_class = MiniMoEConfig

    def __init__(self, config: MiniMoEConfig):
        super().__init__(config)
        self.model = MiniMoETransformer(
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            ff_hidden=config.ff_hidden,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )
        self.regression_head = nn.Linear(config.dim, 1)
        self.init_weights()

    def forward(self, inputs_embeds=None, labels=None, **kwargs):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided (no embedding layer in this model).")
        features = self.model(inputs_embeds)  # (batch, seq, dim)
        logits = self.regression_head(features)  # (batch, seq, 1)
        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)
        return {"loss": loss, "logits": logits}