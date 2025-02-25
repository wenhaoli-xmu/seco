import torch
import torch.utils.checkpoint

from transformers.models.llama.modeling_llama import rotate_half
from functools import partial


def generate_mask(num_query, num_kv, dtype, device):
    mask = torch.full(
        (1, 1, num_query, num_kv), 
        torch.finfo(torch.float32).min, 
        dtype=torch.float32, 
        device=device
    )
    assert num_query <= num_kv
    mask[0,0,:,-num_query:].triu_(diagonal=1)
    mask[0,0,:,:-num_query].fill_(0)
    mask = mask.type(dtype)
    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')
    return mask


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
