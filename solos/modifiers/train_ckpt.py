import torch
import types
import torch.distributed
from transformers.models.llama.modeling_llama import repeat_kv
from ..modifier import Modifier
from .utils import check_and_apply_qk_rope, do_projection
from peft import LoraConfig, get_peft_model, TaskType
from flash_attn import flash_attn_func

from torch.utils.checkpoint import checkpoint
from torch import distributed as dist


def model_forward(self, input_ids, **kwargs):
    """
    Input
    -----
    :input_ids: input indices
    :kv_cache: key value cache
    :kwargs: To absorb useless arguments passed by lib peft
    """
    hidden_states = self.model(input_ids)
    logits = self.lm_head(hidden_states).cpu()
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
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual.to(hidden_states.device) + hidden_states
    
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def self_attn_forward(self, hidden_states):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads


    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim)

    # position embedding
    len1 = self.config.max_position_embeddings if hasattr(self.config, "max_position_embeddings") else 0
    len2 = max(ques.shape[-2], keys.shape[-2])
    cos, sin = self.rotary_emb(keys, seq_len=65536)

    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    attn_output = flash_attn_func(
        q=ques.transpose(-2,-3),
        k=keys.transpose(-2,-3),
        v=vals.transpose(-2,-3),
        causal=True)

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
    

    def forward(self, input_ids, labels):

        # compute logits
        logits = self.model(input_ids=input_ids).to(labels.device)
        
        # compute loss
        logits = logits.squeeze(0)
        labels = labels.squeeze(0)
        return torch.nn.functional.cross_entropy(logits, labels, reduce=False)
