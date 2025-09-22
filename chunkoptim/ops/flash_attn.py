import math

import torch
import triton
import triton.language as tl

from .utils import IS_BF16_ATOM_ADD_SUPPORTED

BLOCK_M = 128
BLOCK_N = 128

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    EVEN_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    if EVEN_M:
        o = tl.load(
            Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        ).to(tl.float32)
        do = tl.load(
            DO
            + off_b * stride_dob
            + off_h * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_d[None, :],
        ).to(tl.float32)

    else:
        o = tl.load(
            Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
            mask=(offs_m[:, None] < seqlen_q),
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            DO
            + off_b * stride_dob
            + off_h * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_d[None, :],
            mask=(offs_m[:, None] < seqlen_q),
            other=0.0,
        ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.jit
def _fwd_kernel(
    Q, K, V,
    Out, Lse,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_ob, stride_oh, stride_om,
    stride_kvb, stride_kvh, stride_kvn,
    nheads, seqlen_q, q_start_idx, headdim,
    seqlen_q_rounded, seqlen_k, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    v_ptrs = V + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    q_idx = q_start_idx + offs_m

    for kv_block_idx in tl.range(0, tl.cdiv(q_start_idx, BLOCK_N) + start_m_block + 1):
        k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = k_idx[:, None] < seqlen_k

        k = tl.load(k_ptrs, mask=kv_mask)
        v = tl.load(v_ptrs, mask=kv_mask)

        qk = tl.dot(q, k.T)
        
        cond = q_idx[:,None] >= k_idx[None,:]
        qk = tl.where(cond, qk, float("-inf"))

        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, m_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

        k_ptrs += stride_kvn * BLOCK_N
        v_ptrs += stride_kvn * BLOCK_N

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i, mask=offs_m < seqlen_q)

    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])

    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _bwd_kernel(
    Q, K, V, DO, DQ, DK, DV,
    LSE, D,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kvb, stride_kvh, stride_kvn,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    nheads, seqlen_q, q_start_idx, headdim,
    seqlen_q_rounded, seqlen_k, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    v_ptrs = V + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + (offs_m[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    dk_ptrs = DK + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    dv_ptrs = DV + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    
    lse_ptrs = LSE + off_hb * seqlen_q_rounded + offs_m
    d_ptrs = D + off_hb * seqlen_q_rounded + offs_m

    mask_m = offs_m < seqlen_q
    if EVEN_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        lse_i = tl.load(lse_ptrs)
        Di = tl.load(d_ptrs)
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse_i = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        Di = tl.load(d_ptrs, mask=mask_m, other=0.0)
        
    dq_block = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    q_idx = q_start_idx + offs_m

    for kv_block_idx in tl.range(0, tl.cdiv(q_start_idx, BLOCK_N) + start_m_block + 1):
        
        k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = k_idx[:, None] < seqlen_k

        k = tl.load(k_ptrs, mask=kv_mask)
        v = tl.load(v_ptrs, mask=kv_mask)

        qk = tl.dot(q, k.T)

        cond = q_idx[:,None] >= k_idx[None,:]
        qk = tl.where(cond, qk, float("-inf"))

        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        dv_block = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_ptrs, dv_block, mask=kv_mask, sem='relaxed')

        dp = tl.dot(do, v.T)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        dk_block = tl.dot(ds.T, q)
        tl.atomic_add(dk_ptrs, dk_block, mask=kv_mask, sem='relaxed')

        dq_block += tl.dot(ds, k)

        k_ptrs += stride_kvn * BLOCK_N
        v_ptrs += stride_kvn * BLOCK_N
        dk_ptrs += stride_kvn * BLOCK_N
        dv_ptrs += stride_kvn * BLOCK_N

    if EVEN_M:
        tl.store(dq_ptrs, dq_block)
    else:
        tl.store(dq_ptrs, dq_block, mask=mask_m[:, None])


def _flash_attn_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale=None):

    global BLOCK_M, BLOCK_N

    batch, seqlen_q, nheads, d = q.shape
    seqlen_k, num_kv_heads, kv_head_dim = k.shape[1:]
    
    assert d <= 128
    assert q.dtype in [torch.float16, torch.bfloat16]
    assert q.is_cuda

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    GROUP_SIZE = nheads // num_kv_heads

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    num_warps = 4 if d <= 64 else 8

    _fwd_kernel[grid](
        q, k, v,
        o, lse,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        o.stride(0), o.stride(2), o.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        nheads, seqlen_q, 0, d,
        seqlen_q_rounded, seqlen_k, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=(seqlen_q % BLOCK_M == 0),
        GROUP_SIZE=GROUP_SIZE,
        num_warps=num_warps,
        num_stages=1)

    return o, lse, softmax_scale


def _flash_attn_backward(
        o: torch.Tensor,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        lse: torch.Tensor,
        softmax_scale: float
):
    if do.stride(-1) != 1:
        do = do.contiguous()

    global BLOCK_M, BLOCK_N

    batch, seqlen_q, nheads, d = q.shape
    seqlen_k, num_kv_heads, kv_head_dim = k.shape[1:]
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    
    delta = torch.empty_like(lse)
    
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    GROUP_SIZE = nheads // num_kv_heads

    grid_preprocess = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    _bwd_preprocess_do_o_dot[grid_preprocess](
        o, do, delta,
        o.stride(0), o.stride(2), o.stride(1),
        do.stride(0), do.stride(2), do.stride(1),
        nheads, seqlen_q, seqlen_q_rounded,
        EVEN_M=(seqlen_q % BLOCK_M == 0),
        BLOCK_M=BLOCK_M, BLOCK_HEADDIM=BLOCK_HEADDIM
    )
    
    num_warps = 4 if kv_head_dim <= 64 else 8

    grid_bwd = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

    _bwd_kernel[grid_bwd](
        q, k, v, do, dq, dk, dv,
        lse, delta,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        do.stride(0), do.stride(2), do.stride(1),
        dq.stride(0), dq.stride(2), dq.stride(1),
        nheads, seqlen_q, 0, d,
        seqlen_q_rounded, seqlen_k, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=(seqlen_q % BLOCK_M == 0),
        GROUP_SIZE=GROUP_SIZE,
        num_warps=num_warps,
        num_stages=1)


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor):

        q = q if q.stride(-1) == 1 else q.contiguous()

        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k ,v, softmax_scale=None)

        ctx.save_for_backward(q, k, v, o, lse)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors

        dq = torch.zeros_like(q)

        if IS_BF16_ATOM_ADD_SUPPORTED:
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
        else:
            dk = torch.zeros_like(k, dtype=torch.float32)
            dv = torch.zeros_like(v, dtype=torch.float32)

        _flash_attn_backward(
            o, do, q, k, v, dq, dk, dv, lse,
            ctx.softmax_scale)
        
        if not IS_BF16_ATOM_ADD_SUPPORTED:
            dk = dk.bfloat16()
            dv = dv.bfloat16()

        return dq, dk, dv, None, None, None

flash_attn_func = FlashAttention.apply