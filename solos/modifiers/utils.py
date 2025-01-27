import torch
import torch.utils.checkpoint

from transformers.models.llama.modeling_llama import rotate_half
from functools import partial


def do_projection(proj, states, num_heads, head_dim):
    return proj(states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)


def apply_rotary_pos_emb(mat, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    mat_embed = (mat * cos) + (rotate_half(mat) * sin)

    return mat_embed


def new_posid(num_token: int, device, dtype, bsz):
    appendix = torch.arange(num_token, device=device)
    appendix = appendix[None,:].expand(bsz, -1)
    return appendix


def check_and_apply_qk_rope(query, key, cos, sin, pos=0):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)
    pos_list = new_posid_spec(pos + num_kv)

    Q = apply_rotary_pos_emb(query, cos, sin, pos_list[:,-num_query:])
    K = apply_rotary_pos_emb(key, cos, sin, pos_list[:,-num_kv:])

    return Q, K
