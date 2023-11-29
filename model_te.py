from typing import Optional

import torch
import torch.nn as nn
from transformer_engine.pytorch import LayerNorm, Linear, LayerNormLinear, LayerNormMLP, DotProductAttention

from lit_gpt.config import Config
from lit_gpt.model import apply_rope, GPT as BaseModel
from lit_gpt.utils import map_old_state_dict_weights


class GPT(BaseModel):
    def __init__(self, config: Config) -> None:
        assert config.padded_vocab_size is not None
        if config._norm_class != "LayerNorm":
            raise NotImplementedError("Llama")

        nn.Module.__init__(self)
        self.config = config

        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict({"h": nn.ModuleList(Block(config, i) for i in range(config.n_layer))})
        self.ln_f_lm_head = LayerNormLinear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias, eps=config.norm_eps
        )

        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, (Linear, LayerNormLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        if isinstance(module, (LayerNormLinear, LayerNormMLP)):
            torch.nn.init.ones_(module.layer_norm_weight)
            torch.nn.init.zeros_(module.layer_norm_bias)
        if isinstance(module, LayerNormMLP):
            torch.nn.init.normal_(module.fc1_weight, mean=0.0, std=0.02)
            if module.fc1_bias is not None:
                torch.nn.init.zeros_(module.fc1_bias)
            torch.nn.init.normal_(module.fc2_weight, mean=0.0, std=0.02)
            if module.fc2_bias is not None:
                torch.nn.init.zeros_(module.fc2_bias)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:
            raise NotImplementedError("Inference")
        cos = self.cos[:T]
        sin = self.sin[:T]

        x = self.wte(idx)
        for block in self.transformer.h:
            x = block(x, cos, sin, input_pos)
        return self.ln_f_lm_head(x)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        mapping = {
            "transformer.wte.weight": "wte.weight",
            "transformer.ln_f.weight": "ln_f_lm_head.layer_norm_weight",
            "transformer.ln_f.bias": "ln_f_lm_head.layer_norm_bias",
            "lm_head.weight": "ln_f_lm_head.weight",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        if config._mlp_class != "GptNeoxMLP":
            raise NotImplementedError("Llama")
        if config.shared_attention_norm:
            raise NotImplementedError

        super().__init__()
        self.config = config

        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.norm_1_attn = LayerNormLinear(config.n_embd, shape, bias=config.bias)
        self.attn = DotProductAttention(
            num_attention_heads=config.n_head,
            kv_channels=config.head_size,
            num_gqa_groups=config.n_query_groups,
            layer_number=block_idx + 1,
        )
        # output projection
        self.proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.norm_2_mlp = LayerNormMLP(hidden_size=config.n_embd, ffn_hidden_size=4 * config.n_embd)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.norm_1_attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        if self.config.n_query_groups != self.config.n_head and self.config.n_query_groups != 1:
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        # flash attn requires (T, B, nh, hs)
        q = q.permute(2, 0, 1, 3)
        k = k.permute(2, 0, 1, 3)
        v = v.permute(2, 0, 1, 3)
        y = self.attn(q, k, v)
        y = y.transpose(0, 1)

        # output projection
        h = self.proj(y)

        if self.config.parallel_residual:
            x = self.norm_2_mlp(x) + h + x
        else:
            x = h + x
            x = self.norm_2_mlp(x) + x
        return x

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        mapping = {
            "norm_1.weight": "norm_1_attn.layer_norm_weight",
            "norm_1.bias": "norm_1_attn.layer_norm_bias",
            "attn.attn.weight": "norm_1_attn.weight",
            "attn.attn.bias": "norm_1_attn.bias",
            "attn.proj.weight": "proj.weight",
            "attn.proj.bias": "proj.bias",
            "norm_2.weight": "norm_2_mlp.layer_norm_weight",
            "norm_2.bias": "norm_2_mlp.layer_norm_bias",
            "mlp.fc.weight": "norm_2_mlp.fc1_weight",
            "mlp.fc.bias": "norm_2_mlp.fc1_bias",
            "mlp.proj.weight": "norm_2_mlp.fc2_weight",
            "mlp.proj.bias": "norm_2_mlp.fc2_bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
