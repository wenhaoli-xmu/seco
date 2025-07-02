import torch
import types
import torch.distributed
from ..modifier import Modifier
from .utils import check_and_apply_qk_rope, do_projection, generate_mask
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from ..page_attn.flash_paged_attn import flash_paged_attn_func
from torch.utils.checkpoint import checkpoint
import math


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def model_forward(self, input_ids, kv_cache, grad_ckpt, **kwargs):
    """
    Input
    -----
    :input_ids: input indices
    :kv_cache: key value cache
    :kwargs: To absorb useless arguments passed by lib peft
    """
    hidden_states = self.model(input_ids, kv_cache, grad_ckpt)
    logits = self.lm_head(hidden_states)
    return logits


def model_model_forward(self, input_ids, kv_cache, grad_ckpt):

    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    for layer in self.layers:
        if grad_ckpt:
            hidden_states = checkpoint(
                layer,
                hidden_states,
                kv_cache,
                use_reentrant=False)
        else:
            hidden_states = layer(
                hidden_states,
                kv_cache)
        
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


def float64_attention(q, k, v, causal=False):
    num_q_heads = q.shape[-2]
    num_kv_heads = k.shape[-2]

    head_dim = q.shape[-1]  # Head dimension
    
    # Expand keys/values if needed for GQA
    if num_q_heads > num_kv_heads:
        expand_factor = num_q_heads // num_kv_heads
        k = k.tile(1, 1, expand_factor, 1)
        v = v.tile(1, 1, expand_factor, 1)
    
    # Compute scaled dot-product attention
    attn_scores = torch.einsum("bqhd, bkhd -> bhqk", q, k) / head_dim**0.5
    
    if causal:
        mask = generate_mask(
            num_query=attn_scores.shape[-2], 
            num_kv=attn_scores.shape[-1], 
            dtype=attn_scores.dtype, 
            device=attn_scores.device)
        attn_scores += mask
    
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v)
    
    return attn_output


def self_attn_forward(self, hidden_states, kv_cache):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    head_dim = embed_dim // num_heads

    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim, head_first=False)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, head_first=False)

    # past length
    past_length = kv_cache.length(self.layer_idx)
    if torch.is_grad_enabled():
        # NOTE: stage-2: second forward prop
        past_length -= ques.shape[1]

    # position embedding
    pos = torch.arange(past_length, past_length + keys.shape[1])
    pos = pos[None, :].to(keys.device)
    cos, sin = self.rotary_emb(keys, pos)

    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    # GQA
    manager = kv_cache[self.layer_idx]

    if not torch.is_grad_enabled():
        # NOTE: stage-1: first forward prop
        manager.update(keys, vals)

    attn_output = flash_paged_attn_func(
        ques,
        keys,
        vals,
        manager)

    attn_output = attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output


class ModelForTraining(Modifier):
    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        self.get_conf(config)
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

        if self.conf['lora']['enable']:
            model = self._init_lora(
                model, 
                lora_rank=self.conf['lora']['r'], 
                lora_alpha=self.conf['lora']['a'], 
                lora_dropout=self.conf['lora']['dropout'])

        super().__init__(model, save_ckp, load_ckp)


    def _init_lora(self, model, lora_rank, lora_alpha, lora_dropout):
        target_modules = r".*\.(self_attn|mlp)\.(q|v)_proj"
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
        params = []
        for layer in self._get_model().model.layers:
            if self.conf['lora']['enable']:
                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.v_proj.lora_A.default.weight,
                    layer.self_attn.v_proj.lora_B.default.weight]
            else:
                params += layer.parameters()
        return params


    def forward(self, input_ids, labels, kv_cache, grad_ckpt=False):

        logits = self.model(
            input_ids=input_ids, 
            kv_cache=kv_cache, 
            grad_ckpt=grad_ckpt).to(input_ids.device)

        if labels is not None:
            logits = logits.to(labels.device)
            logits = logits.squeeze(0)
            labels = labels.squeeze(0)
            return torch.nn.functional.cross_entropy(logits, labels, reduce=False)
        else:
            return logits[:, -1:, :]


    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=128, eos_token_id=[2]):
        raise NotImplementedError
