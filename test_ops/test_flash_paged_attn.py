import torch
from chunkoptim.ops import flash_paged_attn_func
from chunkoptim.cache.kv_cache import CacheManagerSimple
from flash_attn import flash_attn_func


if __name__ == '__main__':
    page_size = 128
    num_heads = 32
    num_kv_heads = 4

    num_kv_cache = 4096
    num_new_toks = 4096

    k_cache = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    v_cache = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    manager = CacheManagerSimple(1, page_size, num_kv_heads, 128)

    q = torch.randn((1, num_new_toks, num_heads, 128), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((1, num_new_toks, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    v = torch.randn((1, num_new_toks, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)

    k_cache.requires_grad_(True)
    v_cache.requires_grad_(True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    q.grad, k.grad, v.grad, k_cache.grad, v_cache.grad = None, None, None, None, None

    ref_bwd = flash_attn_func(q, torch.cat((k_cache, k), dim=1), torch.cat((v_cache, v), dim=1), causal=True)
    ref_bwd.sum().backward()

    ref_grad_q = q.grad.clone()
    ref_grad_k = k.grad.clone()
    ref_grad_v = v.grad.clone()
    ref_grad_kc = k_cache.grad.clone()
    ref_grad_vc = v_cache.grad.clone()

    q.grad, k.grad, v.grad, k_cache.grad, v_cache.grad = None, None, None, None, None
    manager.reset()
    manager.update(k_cache, v_cache)
    manager.update(k, v)
    our_bwd = flash_paged_attn_func(q, k, v, manager)
    our_bwd.sum().backward()

    our_grad_q = q.grad.clone()
    our_grad_k = k.grad.clone()
    our_grad_v = v.grad.clone()

    manager.remove_last_update()
    our_grad_kc, our_grad_vc = manager.grad

    print(torch.dist(ref_bwd, our_bwd))
    print(torch.dist(ref_grad_q, our_grad_q))
    print(torch.dist(ref_grad_k, our_grad_k))
    print(torch.dist(ref_grad_v, our_grad_v))
    print(torch.dist(ref_grad_kc, our_grad_kc))
    print(torch.dist(ref_grad_vc, our_grad_vc))
