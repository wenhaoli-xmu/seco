import torch
import types
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch import distributed as dist
from .utils import do_projection, check_and_apply_qk_rope
from ring_flash_attn import update_ring_flash_attn_params
from ring_flash_attn.adapters.hf_adapter import create_ring_flash_attention_forward, check_params

from ..modifier import Modifier


def model_forward(self, input_ids, **kwargs):
    hidden_states = self.model(input_ids)
    logits = self.lm_head(hidden_states)
    return logits


def model_model_forward(self, input_ids):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    for layer in self.layers:
        hidden_states = checkpoint(
            layer,
            hidden_states,
            use_reentrant=False)
        
    hidden_states = self.norm(hidden_states)
    return hidden_states


def layer_forward(self, hidden_states):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    position_ids = torch.arange(hidden_states.shape[1], dtype=torch.int64, device=hidden_states.device)[None,:]
    q_idx = position_ids.clone().T
    k_idx = position_ids.clone()
    mask = torch.where(q_idx > k_idx, -float('inf'), 0)[None, None, :, :].to(hidden_states.dtype)

    hidden_states = self.self_attn(hidden_states, attention_mask=mask)
    hidden_states = residual.to(hidden_states.device) + hidden_states
    
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def self_attn_forward(self, hidden_states, attention_mask):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    head_dim = embed_dim // num_heads


    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim, head_first=False)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, head_first=False)

    # position embedding
    pos = torch.arange(0, keys.shape[1])
    pos = pos[None, :].to(keys.device)
    cos, sin = self.rotary_emb(keys, pos)
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    attn_output = self.ring_attention(
        query_states=ques,
        key_states=keys,
        value_states=vals,
        attention_mask=attention_mask,
        query_length=ques.shape[1],
        is_causal=True)

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
            ring_attention = create_ring_flash_attention_forward(None, 1)[0]
            layer.self_attn.ring_attention = lambda *args, **kwargs: ring_attention(*args, **kwargs)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

        super().__init__(model, save_ckp, load_ckp)


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
    

    def forward(self, input_ids, labels):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        cu_seqlens = torch.tensor([0, input_ids.shape[-1]], dtype=torch.int32, device=rank)
        update_ring_flash_attn_params(cu_seqlens, None)

        input_ids_chunk = torch.chunk(input_ids, world_size, dim=1)[rank]
        logits_chunk = self.model(input_ids=input_ids_chunk)
        labels_chunk = torch.chunk(labels, world_size, dim=1)[rank]

        loss = F.cross_entropy(
            logits_chunk.view(-1, logits_chunk.shape[-1]), 
            labels_chunk.view(-1),
            reduction='mean')
        
        return loss

