import torch
import types
import torch.distributed
from transformers.models.llama.modeling_llama import repeat_kv
from ..modifier import Modifier
from .utils import ScoreHead, check_and_apply_qk_rope, prune_labels, merge, do_projection
from copy import deepcopy


def model_forward(self, input_ids, enable_prune):
    hidden_states, mask = self.model(input_ids, enable_prune)
    logits = self.lm_head(hidden_states).float()
    return logits, mask



def model_model_forward(self, input_ids, enable_prune):

    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    mask = None

    for layer in self.layers:
        if hasattr(layer, 'score_head') and enable_prune:
            score = layer.score_head(hidden_states)
            mask = (score.sigmoid() > torch.rand_like(score)).ravel().tolist()
            mask[-self.last_n:] = [False] * self.last_n
            hidden_states = merge(hidden_states, mask, self.merge_method)

        hidden_states = layer(hidden_states)
        
    hidden_states = self.norm(hidden_states)

    return hidden_states, mask



def layer_forward(self, hidden_states):    

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states
    
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
    cos, sin = self.rotary_emb(keys, seq_len=max(len1, len2))
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)


    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query=ques,
        key=keys,
        value=vals,
        is_causal=True)

    attn_output = attn_output.transpose(1,2).flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output


class ModelForEvaluation(Modifier):
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
    def forward(self, input_ids, labels, enable_prune=True):

        """
        Return
        ------
        1. mask == None: return (loss, score)
        2. mask != None: return (loss, ratio, reward)

        :loss: language modeling loss
        :ratio: masked out ratio
        :score: output score of mask predictor
        :reward: used in policy gradient
        """

        ratio = None
        last_n = self.conf['last_n']
        
        # preprocess mask and labels
        labels[:, :-last_n] = -100

        logits, mask = self.model(input_ids=input_ids, enable_prune=enable_prune)

        if mask is not None:
            mask[-last_n:] = [False] * last_n
            labels = prune_labels(labels, mask)
            ratio = sum(mask) / (len(mask) - last_n)
        
        # compute loss
        logits = logits.squeeze(0)
        labels = labels.squeeze(0)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return dict(loss=loss, ratio=ratio, mask=mask)
