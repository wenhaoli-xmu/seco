import torch
import types
import torch.distributed
from transformers.models.llama.modeling_llama import repeat_kv
from ..modifier import Modifier
from .utils import ScoreHead, check_and_apply_qk_rope, merge, do_projection
from flash_attn import flash_attn_func


def model_forward(self, input_ids, kv_cache, enable_prune):
    hidden_states, kv_cache, mask = self.model(
        input_ids, 
        kv_cache, 
        enable_prune)

    logits = self.lm_head(hidden_states[:,-1:,:]).float()
    return logits, kv_cache, mask



def model_model_forward(self, input_ids, kv_cache, enable_prune):

    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    mask = None

    if kv_cache is None:
        kv_cache = [[None, None] for _ in range(len(self.layers))]

    for layer in self.layers:
        if hasattr(layer, 'score_head') and enable_prune:
            score = layer.score_head(hidden_states)
            mask = (score.sigmoid() > torch.rand_like(score)).ravel().tolist()
            hidden_states = merge(hidden_states, mask, self.merge_method)

        hidden_states, kv_cache = layer(hidden_states, kv_cache)
        
    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache, mask



def layer_forward(self, hidden_states, kv_cache):

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kv_cache = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual + hidden_states
    
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache


def self_attn_forward(self, hidden_states, kv_cache):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    head_dim = embed_dim // num_heads
    num_groups = num_heads // num_kv_heads


    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim)

    # position embedding
    len1 = self.config.max_position_embeddings if hasattr(self.config, "max_position_embeddings") else 0
    len2 = max(ques.shape[-2], keys.shape[-2])
    cos, sin = self.rotary_emb(keys, seq_len=max(len1, len2))
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    if kv_cache[self.layer_idx][0] is not None:
        keys = torch.cat([kv_cache[self.layer_idx][0], keys], dim=-2)
        vals = torch.cat([kv_cache[self.layer_idx][1], vals], dim=-2)
    kv_cache[self.layer_idx][0] = keys.data.clone()
    kv_cache[self.layer_idx][1] = vals.data.clone()

    keys = repeat_kv(keys, num_groups)
    vals = repeat_kv(vals, num_groups)

    ques = ques.transpose(1,2)
    keys = keys.transpose(1,2)
    vals = vals.transpose(1,2)

    attn_output = flash_attn_func(
        q=ques,
        k=keys,
        v=vals,
        causal=True)

    attn_output = attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output, kv_cache


class ModelForGeneration(Modifier):
    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        self.get_conf(config)
        model.forward = types.MethodType(model_forward, model)
        model.model.forward = types.MethodType(model_model_forward, model.model)
        model.model.merge_method = self.conf['merge_method']
        model.model.last_n = self.conf['last_n']

        for i, layer in enumerate(model.model.layers):
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            if i == self.conf['layer_cut']:
                layer.score_head = ScoreHead(hidden_size=model.config.hidden_size, kernel_size=2)

        super().__init__(model, save_ckp, load_ckp)


    def get_model(self):
        return self.model


    def ft_params(self):
        params = []
        for i, layer in enumerate(self.get_model().model.layers):
            if i == self.conf['layer_cut']:
                params += layer.score_head.parameters()
        return params
    

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, eos_token_id=[2], enable_prune=True):
        input_ids = input_ids.cuda()

        # prefilling
        logit, kv_cache, mask = self.model(
            input_ids=input_ids, 
            kv_cache=None,
            enable_prune=enable_prune)

        new_tok = logit.argmax(dim=-1)
        new_ids = [new_tok]

        # generation
        while len(new_ids) < max_new_tokens:

            logit, kv_cache, _ = self.model(
                input_ids=new_tok, 
                kv_cache=kv_cache, 
                enable_prune=False)

            new_tok = logit.argmax(dim=-1)
            if new_tok.ravel().item() in eos_token_id: break
            new_ids.append(new_tok.ravel().item())

        new_ids = torch.tensor(new_ids, dtype=input_ids.dtype, device='cuda')[None, :]
        return torch.cat([input_ids, new_ids], dim=-1), mask
