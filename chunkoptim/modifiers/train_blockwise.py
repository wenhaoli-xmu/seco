import torch
import types
import torch.distributed
from ..modifier import Modifier
from .utils import check_and_apply_qk_rope, do_projection, generate_mask

import torch.nn.functional as F
from ..ops.flash_paged_attn import flash_paged_attn_func
from torch.utils.checkpoint import checkpoint


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

    # =========================================
    stage = 2 if torch.is_grad_enabled() else 1
    kv_cache.visit(self.layer_idx)
    # =========================================

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    head_dim = embed_dim // num_heads

    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim, head_first=False)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, head_first=False)

    # ===========================================
    past_length = kv_cache[self.layer_idx].num_kv
    if stage == 2:
        past_length -= ques.shape[1]
    # ===========================================

    # position embedding
    pos = torch.arange(past_length, past_length + keys.shape[1])
    pos = pos[None, :].to(keys.device)
    cos, sin = self.rotary_emb(keys, pos)
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    # ================================================
    kv_cache[self.layer_idx].update(keys, vals, stage)
    # ================================================

    attn_output = flash_paged_attn_func(
        ques,
        keys,
        vals,
        kv_cache[self.layer_idx])

    attn_output = attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output


class ModelForTraining(Modifier):
    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        self.get_conf(config)
        model.forward = types.MethodType(model_forward, model)
        model.model.forward = types.MethodType(model_model_forward, model.model)
        self.num_layers = len(model.model.layers)

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
        for layer in self.model.model.layers:
            params.extend([
                layer.self_attn.q_proj.weight,
                layer.self_attn.k_proj.weight,
                layer.self_attn.v_proj.weight,
                layer.self_attn.o_proj.weight,
                layer.mlp.gate_proj.weight,
                layer.mlp.up_proj.weight,
                layer.mlp.down_proj.weight])
        params.append(self.model.lm_head.weight)
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
