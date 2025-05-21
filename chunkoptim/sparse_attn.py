import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
import triton
import triton.language as tl


@triton.jit
def gqa_matmul_kernel(
        q_ptr, 
        k_ptr, 
        s_ptr,
        QH, KH, M, N, K,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_sb, stride_sh, stride_sn,
        TILE_M: tl.constexpr, 
        TILE_N: tl.constexpr, 
        TILE_K: tl.constexpr,
):
    mn_idx, bh_idx = tl.program_id(axis=0), tl.program_id(axis=1)
    
    m_idx = mn_idx // tl.cdiv(N, TILE_N)
    n_idx = mn_idx % tl.cdiv(N, TILE_N)

    b_idx = bh_idx // QH
    qh_idx = bh_idx % QH
    kh_idx = qh_idx % KH

    m_rng = m_idx * TILE_M + tl.arange(0, TILE_M)
    n_rng = n_idx * TILE_N + tl.arange(0, TILE_N)
    k_rng = tl.arange(0, TILE_K)

    q_off = b_idx * stride_qb + qh_idx * stride_qh + m_rng[:, None] * stride_qm + k_rng[None, :]
    k_off = b_idx * stride_kb + kh_idx * stride_kh + n_rng[:, None] * stride_kn + k_rng[None, :]
    s_off = b_idx * stride_sb + kh_idx * stride_sh + n_rng * stride_sn

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    m_msk = m_rng < M
    n_msk = n_rng < N

    for _ in range(0, tl.cdiv(K, TILE_K)):
        q = tl.load(q_ptr + q_off, mask=m_msk[:, None])
        k = tl.load(k_ptr + k_off, mask=n_msk[:, None])
        acc = tl.dot(q, k.T, acc)
        q_ptr += TILE_K
        k_ptr += TILE_K

    acc = tl.sum(acc, axis=0)

    tl.atomic_add(
        s_ptr + s_off,
        acc,
        n_msk)
    

def gqa_matmul(q, k):
    TILE_M = 128
    TILE_N = 128
    TILE_K = 16
    
    B, M, QH, K = q.shape
    _, N, KH, _ = k.shape

    assert k.shape[0] == B and k.shape[3] == K
    assert K % TILE_K == 0

    s = torch.empty((B, N, KH), dtype=torch.float32, device=k.device)
    s.zero_()

    grid = (
        triton.cdiv(M, TILE_M) * triton.cdiv(N, TILE_N), 
        B * QH)

    gqa_matmul_kernel[grid](
        q,
        k,
        s,
        QH, KH, M, N, K,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        s.stride(0), s.stride(2), s.stride(1),
        TILE_M,
        TILE_N,
        TILE_K)
    
    return s


def torch_gqa_matmul(q, k):
    head_group = q.shape[2] // k.shape[2]
    q = q.transpose(-2,-3)
    k = k.transpose(-2,-3)
    k = k.repeat(1, head_group, 1, 1)
    return q @ k.transpose(-1,-2)


class SparseBackwardAttn(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            q, 
            k, 
            v, 
            kv_cache, 
            layer_idx, 
            attention_mask=None):
        
        softmax_scale = q.shape[-1] ** (-0.5)

        out, _, _, _, out_padded, softmax_lse, _, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1,-1),
            alibi_slopes=None,
            return_softmax=False)
        
        # do_sparse_backward = False
        # indices = None
        past_len = k.shape[1] - q.shape[1]

        # if sparse_backward and past_len > 0:
        #     budget = int(past_len * ratio_lowerbound)
        #     budget = max(budget, budget_lowerbound)
        #     if past_len > budget:
        #         score = gqa_matmul(q, k)
        #         indices = torch.topk(
        #             score[:, : past_len, :], 
        #             k=budget, dim=1).indices
        #         indices = indices.unsqueeze(-1
        #             ).expand(-1,-1,-1,k.shape[-1])
        #         do_sparse_backward = True

        ctx.past_len = past_len
        # ctx.input_shape = (k.shape, v.shape)
        ctx.softmax_scale = softmax_scale
        ctx.kv_cache = kv_cache
        ctx.layer_idx = layer_idx
        # ctx.do_sparse_backward = do_sparse_backward
        ctx.save_for_backward(q, out_padded, softmax_lse, rng_state)

        return out


    @staticmethod
    def backward(ctx, dout):
        q, out_padded, softmax_lse, rng_state = ctx.saved_tensors
        k, v = ctx.kv_cache.gather(ctx.layer_idx)
        # k_shape, v_shape = ctx.input_shape

        # _, _, _, _, out_padded, softmax_lse, _, rng_state = _flash_attn_forward(
        #     q,
        #     k,
        #     v,
        #     dropout_p=0.0,
        #     softmax_scale=ctx.softmax_scale,
        #     causal=True,
        #     window_size=(-1, -1),
        #     alibi_slopes=None,
        #     return_softmax=False)

        # if ctx.do_sparse_backward:
        #     past_k = k[:, : ctx.past_len, ...]
        #     past_v = v[:, : ctx.past_len, ...]
        #     recent_k = k[:, ctx.past_len:, ...]
        #     recent_v = v[:, ctx.past_len:, ...]
        #     past_k = torch.gather(past_k, dim=1, index=indices)
        #     past_v = torch.gather(past_v, dim=1, index=indices)
        #     k = torch.cat([past_k, recent_k], dim=1).contiguous()
        #     v = torch.cat([past_v, recent_v], dim=1).contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out_padded,
            softmax_lse,
            dq,
            dk,
            dv,
            0.0,
            ctx.softmax_scale,
            True,
            (-1, -1),
            None,
            False,
            rng_state)
        
        dq = dq[..., : dout.shape[-1]]
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        
        # if ctx.do_sparse_backward:
        #     dk = torch.zeros(k_shape, device=q.device, dtype=q.dtype
        #         ).scatter_(dim=1, index=indices, src=dk)
            
        #     dv = torch.zeros(v_shape, device=q.device, dtype=q.dtype
        #         ).scatter_(dim=1, index=indices, src=dv)

        return dq, dk, dv, None, None, None


flash_attn_func = SparseBackwardAttn.apply