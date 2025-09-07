import torch
from chunkoptim.cache.topk_cache import SparseCacheManager
from chunkoptim.ops.flash_paged_topk import flash_paged_sparse_attn_func
from flash_attn import flash_attn_func


if __name__ == '__main__':

    from profiler import WallTime

    profile_topk = WallTime('topk', cuda=0)
    profile_fwd = WallTime('fwd', cuda=0)
    profile_sparse = WallTime('sparse', cuda=0)
    profile_dense = WallTime('dense', cuda=0)

    page_size = 64
    batch_size = 1
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128

    num_kv_cache = 4096
    num_new_toks = 4096

    assert num_new_toks % page_size == 0, "num_new_toks must be a multiple of page_size"

    total_kv_len = num_kv_cache + num_new_toks
    num_kv_blocks = (total_kv_len + page_size - 1) // page_size

    num_selected_blocks = 128
    num_query_blocks = num_new_toks // page_size

    k_cache = torch.randn((batch_size, num_kv_cache, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    v_cache = torch.randn((batch_size, num_kv_cache, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    
    q = torch.randn((batch_size, num_new_toks, num_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    k_new = torch.randn((batch_size, num_new_toks, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    v_new = torch.randn((batch_size, num_new_toks, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    
    full_key = torch.cat([k_cache, k_new], dim=1)
    full_val = torch.cat([v_cache, v_new], dim=1)

    for _ in range(3):
        with WallTime.get('dense'):
            out_dense = flash_attn_func(q, full_key, full_val, causal=True)

    manager_sparse = SparseCacheManager(
        batch_size=batch_size, 
        page_size=page_size, 
        num_kv_heads=num_kv_heads, 
        head_dim=head_dim, 
        sparse_topk=num_selected_blocks
    )
    manager_sparse.update(k_cache, v_cache)
    manager_sparse.update(k_new, v_new)

    for _ in range(3):
        with WallTime.get('sparse'):
            out_sparse = flash_paged_sparse_attn_func(q, None, None, manager_sparse)

    test_q_head = 22
    test_q_page = 0

    kv_groups = num_heads // num_kv_heads
    sparse_key = []
    sparse_val = []
    last_q = q[:1, test_q_page: test_q_page + 1, test_q_head: test_q_head + 1, :]

    indices_list = manager_sparse.top_indices[0, test_q_head, test_q_page].tolist()
    indices_list.sort()
    for idx in indices_list:
        key_page = full_key[:, page_size * idx: page_size * (idx + 1), (test_q_head//kv_groups): (test_q_head//kv_groups) + 1]
        val_page = full_val[:, page_size * idx: page_size * (idx + 1), (test_q_head//kv_groups): (test_q_head//kv_groups) + 1]
        sparse_key.append(key_page)
        sparse_val.append(val_page)
    sparse_key = torch.cat(sparse_key, dim=1)
    sparse_val = torch.cat(sparse_val, dim=1)

    out_sparse_2 = flash_attn_func(last_q, sparse_key, sparse_val, causal=True)

    print(f"Output difference: {torch.dist(out_dense, out_sparse)}")
    print(f"Output difference (sparse simulate): {torch.dist(out_sparse_2, out_sparse[:, test_q_page: test_q_page + 1, test_q_head: test_q_head + 1, :])}")

    WallTime.get('topk').result(detail=True)
    WallTime.get('fwd').result(detail=True)
    WallTime.get('sparse').result(detail=True)
    WallTime.get('dense').result(detail=True)