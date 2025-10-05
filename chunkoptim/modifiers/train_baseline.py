import torch
import types
import torch.distributed
from ..modifier import Modifier
from .utils import check_and_apply_qk_rope, do_projection, generate_mask

from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from ..ops import flash_attn_func


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


def self_attn_forward(self, hidden_states, kv_cache):

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

    attn_output = flash_attn_func(ques, keys, vals)

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

        # compute logits
        logits = self.model(input_ids=input_ids, kv_cache=kv_cache, grad_ckpt=grad_ckpt).to(labels.device)
        
        # compute loss
        logits = logits.squeeze(0)
        labels = labels.squeeze(0)
        return torch.nn.functional.cross_entropy(logits, labels, reduce=False)
