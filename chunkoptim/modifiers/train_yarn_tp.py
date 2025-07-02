import torch
import types
import torch.nn as nn
import torch.distributed as dist
from ..modifier import Modifier
from .utils import check_and_apply_qk_rope
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from ..ops.flash_paged_attn import flash_paged_attn_func
from ..ops.all_gather import _AllGather
from torch.utils.checkpoint import checkpoint
import math


def get_tensor_parallel_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_tensor_parallel_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class LlamaYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn(device)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(x.dtype), persistent=False)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mscale = float(_yarn_get_mscale(self.scale) * self.attn_factor)


class ColumnParallelLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.world_size = get_tensor_parallel_world_size()
        self.rank = get_tensor_parallel_rank()

        output_size_per_partition = linear_layer.out_features // self.world_size
        self.weight = nn.Parameter(
            linear_layer.weight.data[
                self.rank * output_size_per_partition: (self.rank + 1) * output_size_per_partition, :
            ].clone()
        )
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(
                linear_layer.bias.data[
                    self.rank * output_size_per_partition: (self.rank + 1) * output_size_per_partition
                ].clone()
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output_parallel = F.linear(x, self.weight, self.bias)
        return output_parallel


class RowParallelLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.world_size = get_tensor_parallel_world_size()
        self.rank = get_tensor_parallel_rank()

        input_size_per_partition = linear_layer.in_features // self.world_size
        self.weight = nn.Parameter(
            linear_layer.weight.data[
                :, self.rank * input_size_per_partition: (self.rank + 1) * input_size_per_partition
            ].clone()
        )
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output_parallel = F.linear(x, self.weight)
        
        if self.world_size > 1:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
    
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        return output_parallel
    

def model_forward(self, input_ids, kv_cache, grad_ckpt, **kwargs):
    hidden_states = self.model(input_ids, kv_cache, grad_ckpt)
    logits = self.lm_head(hidden_states)
    return logits


def model_model_forward(self, input_ids, kv_cache, grad_ckpt):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    for layer in self.layers:
        if grad_ckpt:
            hidden_states = checkpoint(layer, hidden_states, kv_cache, use_reentrant=False)
        else:
            hidden_states = layer(hidden_states, kv_cache)
        
    hidden_states = self.norm(hidden_states)
    return hidden_states


def layer_forward(self, hidden_states, kv_cache):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual.to(hidden_states.device) + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def self_attn_forward(self, hidden_states, kv_cache):
    world_size = get_tensor_parallel_world_size()
    
    num_heads = self.config.num_attention_heads // world_size
    num_kv_heads = self.config.num_key_value_heads // world_size
    embed_dim = self.config.hidden_size
    head_dim = embed_dim // self.config.num_attention_heads

    ques = self.q_proj(hidden_states)
    keys = self.k_proj(hidden_states)
    vals = self.v_proj(hidden_states)

    ques = ques.view(ques.shape[0], ques.shape[1], num_heads, head_dim)
    keys = keys.view(keys.shape[0], keys.shape[1], num_kv_heads, head_dim)
    vals = vals.view(vals.shape[0], vals.shape[1], num_kv_heads, head_dim)

    past_length = kv_cache.length(self.layer_idx)
    if torch.is_grad_enabled():
        past_length -= ques.shape[1]

    pos = torch.arange(past_length, past_length + keys.shape[1], device=keys.device).unsqueeze(0)
    cos, sin = self.rotary_emb(vals, pos)
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    manager = kv_cache[self.layer_idx]
    if not torch.is_grad_enabled():
        manager.update(keys, vals)

    attn_output = flash_paged_attn_func(ques, keys, vals, manager)
    attn_output = attn_output.flatten(2)
    
    attn_output = self.o_proj(attn_output)

    return attn_output

def mlp_forward(self, hidden_state):
    gate_output = self.gate_proj(hidden_state)
    up_output = self.up_proj(hidden_state)
    
    # 逐元素相乘
    intermediate = self.act_fn(gate_output) * up_output
    
    return self.down_proj(intermediate)

class ModelForTraining(Modifier):
    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        self.get_conf(config)
        
        self._parallelize_model(model)

        model.forward = types.MethodType(model_forward, model)
        model.model.forward = types.MethodType(model_model_forward, model.model)
        self.num_layers = len(model.model.layers)

        model.model.rotary_emb = LlamaYaRNScaledRotaryEmbedding(
            dim=model.config.hidden_size // model.config.num_attention_heads,
            max_position_embeddings=self.conf['extend_to'], # 4M
            scale=self.conf['extend_to'] // model.config.max_position_embeddings,
            original_max_position_embeddings=model.config.max_position_embeddings)

        for layer in model.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)

        if self.conf['lora']['enable']:
            model = self._init_lora(
                model, 
                lora_rank=self.conf['lora']['r'], 
                lora_alpha=self.conf['lora']['a'], 
                lora_dropout=self.conf['lora']['dropout'])

        super().__init__(model, save_ckp, load_ckp)

    def _parallelize_model(self, model):
        world_size = get_tensor_parallel_world_size()
        if world_size <= 1:
            return

        for layer in model.model.layers:
            # Attention
            layer.self_attn.q_proj = ColumnParallelLinear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = ColumnParallelLinear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = ColumnParallelLinear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = RowParallelLinear(layer.self_attn.o_proj)
            # MLP
            layer.mlp.gate_proj = ColumnParallelLinear(layer.mlp.gate_proj)
            layer.mlp.up_proj = ColumnParallelLinear(layer.mlp.up_proj)
            layer.mlp.down_proj = RowParallelLinear(layer.mlp.down_proj)

        model.lm_head = ColumnParallelLinear(model.lm_head)

    def _init_lora(self, model, lora_rank, lora_alpha, lora_dropout):
        target_modules = r".*\.(self_attn|mlp)\.(q_proj|v_proj|gate_proj|up_proj)"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules)
        return get_peft_model(model, peft_config)

    def _get_model(self):
        if self.conf['lora']['enable']:
            return self.model.model
        else:
            return self.model

    def ft_params(self):
        return self.model.parameters()

    def forward(self, input_ids, labels, kv_cache, grad_ckpt=False):
        logits = self.model(
            input_ids=input_ids, 
            kv_cache=kv_cache, 
            grad_ckpt=grad_ckpt)
        logits = _AllGather.apply(logits)

        if labels is not None:
            logits = logits.to(labels.device)
            logits = logits.squeeze(0)
            labels = labels.squeeze(0)
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        else:
            return logits[:, -1:, :]

    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=128, eos_token_id=[2]):
        raise NotImplementedError
