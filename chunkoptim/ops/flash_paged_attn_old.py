"""
*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

import math

import torch
import triton
import triton.language as tl
from .utils import IS_BF16_ATOM_ADD_SUPPORTED

@triton.jit
def _fwd_kernel(
    Q,
    T,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kvb,
    stride_kvh,
    stride_kvn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    kv_offs = off_b * stride_kvb + (off_h // GROUP_SIZE) * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])

    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M & EVEN_N:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    for n_idx in range(tl.cdiv(seqlen_k, BLOCK_N)):
        
        k_page_ptr = tl.load(T + n_idx * 4)
        v_page_ptr = tl.load(T + n_idx * 4 + 1)

        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))

        k_ptrs = k_page_ptr + kv_offs
        v_ptrs = v_page_ptr + kv_offs

        start_n = n_idx * BLOCK_N

        if EVEN_N & EVEN_M:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(
                k_ptrs,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k.T)

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(seqlen_k - seqlen_q + offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float('-inf'))

        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)

        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(
                v_ptrs,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0)

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)

    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)


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
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)

    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    seqlen_k,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    if EVEN_N & EVEN_M:
        tl.atomic_add(dv_ptrs, dv)
        tl.atomic_add(dk_ptrs, dk)
    else:
        tl.atomic_add(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
        tl.atomic_add(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kvn,
    stride_dom,
    stride_dqm,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    begin_m = 0

    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    kv_offs = tl.arange(0, BLOCK_N)[:, None] * stride_kvn + offs_d[None, :]

    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + kv_offs
    v_ptrs = V + kv_offs

    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])

    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    if begin_m >= seqlen_q:
        dk_ptrs = DK + kv_offs
        dv_ptrs = DV + kv_offs

        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            seqlen_k,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
        )
        return

    if EVEN_N & EVEN_M:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)

    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        if EVEN_M:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)

        qk = tl.dot(q, k.T)
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(seqlen_k - seqlen_q + offs_m_curr[:, None] >= offs_n[None, :], qk, float("-inf"))

        if not EVEN_M:
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        if EVEN_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0)

        dv += tl.dot(p.to(do.dtype).T, do)

        if not EVEN_M:
            tl.debug_barrier()
        dp = tl.dot(do, v.T)

        Di = tl.load(D + offs_m_curr)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        dk += tl.dot(ds.T, q)

        if not EVEN_M: 
            tl.debug_barrier()

        if not ATOMIC_ADD:
            if EVEN_M:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                dq = tl.load(
                    dq_ptrs,
                    mask=offs_m_curr[:, None] < seqlen_q,
                    other=0.0,
                    eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(
                    dq_ptrs,
                    dq,
                    mask=offs_m_curr[:, None] < seqlen_q,
                    eviction_policy="evict_last")

        else:
            dq = tl.dot(ds, k)
            if EVEN_M:
                tl.atomic_add(dq_ptrs, dq)
            else:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)

        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

    dk_ptrs = DK + kv_offs
    dv_ptrs = DV + kv_offs

    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        seqlen_k,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.jit
def _bwd_kernel(
    Q,
    DO,
    DQ,
    T,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kvb,
    stride_kvh,
    stride_kvn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    IS_BF16_ATOM_ADD_SUPPORTED: tl.constexpr
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    Q += off_b * stride_qb + off_h * stride_qh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh

    # batch-size offset, head offset
    kv_offs = off_b * stride_kvb + (off_h // GROUP_SIZE) * stride_kvh

    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):

            k_page_ptr = tl.load(T + start_n * 4)
            v_page_ptr = tl.load(T + start_n * 4 + 1)
            dk_page_ptr = tl.load(T + start_n * 4 + 2)
            dv_page_ptr = tl.load(T + start_n * 4 + 3)

            k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
            v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))
            dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))
            dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))

            k_ptrs = k_page_ptr + kv_offs
            v_ptrs = v_page_ptr + kv_offs
            dk_ptrs = dk_page_ptr + kv_offs
            dv_ptrs = dv_page_ptr + kv_offs

            _bwd_kernel_one_col_block(
                start_n,
                Q,
                k_ptrs,
                v_ptrs,
                DO,
                DQ,
                dk_ptrs,
                dv_ptrs,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kvn,
                stride_dom,
                stride_dqm,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N)
    else:
        start_n = tl.program_id(0)
        
        k_page_ptr = tl.load(T + start_n * 4)
        v_page_ptr = tl.load(T + start_n * 4 + 1)
        dk_page_ptr = tl.load(T + start_n * 4 + 2)
        dv_page_ptr = tl.load(T + start_n * 4 + 3)

        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))
        dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))
        dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))

        k_ptrs = k_page_ptr + kv_offs
        v_ptrs = v_page_ptr + kv_offs
        dk_ptrs = dk_page_ptr + kv_offs
        dv_ptrs = dv_page_ptr + kv_offs

        _bwd_kernel_one_col_block(
            start_n,
            Q,
            k_ptrs,
            v_ptrs,
            DO,
            DQ,
            dk_ptrs,
            dv_ptrs,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kvn,
            stride_dom,
            stride_dqm,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N)


def _flash_attn_forward(
        q: torch.Tensor, 
        page_table: torch.Tensor,
        num_kv: int, 
        page_size: int, 
        num_kv_heads: int, 
        kv_head_dim: int, 
        bias=None, 
        causal=False, 
        softmax_scale=None):

    batch, seqlen_q, nheads, d = q.shape

    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / page_size) * page_size
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = page_size
    GROUP_SIZE = nheads // num_kv_heads

    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    seqlen_k = num_kv

    _fwd_kernel[grid](
        q,
        page_table,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        page_size * num_kv_heads * kv_head_dim,
        kv_head_dim,
        num_kv_heads * kv_head_dim,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        causal,
        BLOCK_HEADDIM,
        EVEN_M=seqlen_q % BLOCK == 0,
        EVEN_N=seqlen_k % BLOCK == 0,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=num_warps,
        num_stages=1)

    return o, lse, softmax_scale  # softmax_scale could have been updated


def _flash_attn_backward(
        o: torch.Tensor, 
        do: torch.Tensor, 
        q: torch.Tensor, 
        dq: torch.Tensor, 
        page_table: torch.Tensor,
        num_kv: int, 
        page_size: int, 
        num_kv_heads: int, 
        kv_head_dim: int, 
        lse: torch.Tensor, 
        bias=None, 
        causal=False, 
        softmax_scale=None,
):
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
 
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / page_size) * page_size
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    if not IS_BF16_ATOM_ADD_SUPPORTED:
        # Convert dq to float32 for backward pass
        dq_accum = torch.zeros_like(q, dtype=torch.float32)

    delta = torch.empty_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    GROUP_SIZE = nheads // num_kv_heads
    BLOCK = page_size

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=BLOCK,
        BLOCK_HEADDIM=BLOCK_HEADDIM)

    grid = lambda META: (
        triton.cdiv(num_kv, BLOCK) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,)
    
    seqlen_k = num_kv

    _bwd_kernel[grid](
        q,
        do,
        dq if IS_BF16_ATOM_ADD_SUPPORTED else dq_accum,
        page_table,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        page_size * num_kv_heads * kv_head_dim,
        kv_head_dim,
        num_kv_heads * kv_head_dim,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq.stride(0),
        dq.stride(2),
        dq.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        causal,
        EVEN_M=seqlen_q % BLOCK == 0,
        EVEN_N=seqlen_k % BLOCK == 0,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        GROUP_SIZE=GROUP_SIZE,
        IS_BF16_ATOM_ADD_SUPPORTED=IS_BF16_ATOM_ADD_SUPPORTED,
        SEQUENCE_PARALLEL=True,
        num_warps=8,
        num_stages=1)

    if not IS_BF16_ATOM_ADD_SUPPORTED:
        # Convert dq back to bf16
        dq.copy_(dq_accum)


class FlashPagedAttn(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            manager):

        q = q if q.stride(-1) == 1 else q.contiguous()

        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, manager.page_table,
            manager.num_kv, manager.page_size, manager.num_kv_heads, manager.head_dim,
            bias=None, causal=True, softmax_scale=None)
        
        ctx.save_for_backward(q, o, lse)
        ctx.manager = manager

        return o

    @staticmethod
    def backward(ctx, do):
        q, o, lse = ctx.saved_tensors

        with torch.inference_mode():
            dq = torch.zeros_like(q)

            _flash_attn_backward(
                o, do, q, dq,
                ctx.manager.page_table,
                ctx.manager.num_kv, 
                ctx.manager.page_size, 
                ctx.manager.num_kv_heads, 
                ctx.manager.head_dim,
                lse=lse, 
                bias=None, 
                causal=True, 
                softmax_scale=ctx.softmax_scale)
            
            dk, dv = ctx.manager.grad
            
        return dq, dk, dv, None


flash_paged_attn_func = FlashPagedAttn.apply


if __name__ == '__main__':
    from flash_attn import flash_attn_func
    from chunkoptim.kv_cache import CacheManager
    page_size = 64

    num_heads = 32
    num_kv_heads = 4

    num_kv_cache = 12800
    num_new_toks = 1024

    k_cache1 = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    v_cache1 = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    k_cache2 = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    v_cache2 = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    manager = CacheManager(1, page_size, num_kv_heads, 128)

    q = torch.randn((1, num_new_toks, num_heads, 128), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((1, num_new_toks, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    v = torch.randn((1, num_new_toks, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)

    k_cache1.requires_grad_(True)
    v_cache1.requires_grad_(True)
    k_cache2.requires_grad_(True)
    v_cache2.requires_grad_(True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    from profiler import WallTime
    ref_time_bwd = WallTime('ref-bwd', cuda=0)
    our_time_bwd = WallTime('our-bwd', cuda=0)

    for _ in range(20):

        q.grad, k.grad, v.grad, k_cache1.grad, v_cache1.grad, k_cache2.grad, v_cache2.grad = None, None, None, None, None, None, None
        ref_bwd = flash_attn_func(q, torch.cat((k_cache1, k_cache2, k), dim=1), torch.cat((v_cache1, v_cache2, v), dim=1), causal=True)

        with ref_time_bwd:
            ref_bwd.sum().backward()

        ref_grad_q = q.grad.clone()
        ref_grad_k = k.grad.clone()
        ref_grad_v = v.grad.clone()
        ref_grad_kc1 = k_cache1.grad.clone()
        ref_grad_vc1 = v_cache1.grad.clone()
        ref_grad_kc2 = k_cache2.grad.clone()
        ref_grad_vc2 = v_cache2.grad.clone()

        q.grad, k.grad, v.grad, k_cache1.grad, v_cache1.grad, k_cache2.grad, v_cache2.grad = None, None, None, None, None, None, None
        manager.reset()
        manager.update(k_cache1, v_cache1)
        manager.update(k_cache2, v_cache2)
        manager.update(k, v)
        our_bwd = flash_paged_attn_func(q, k, v, manager)
        
        with our_time_bwd:
            our_bwd.sum().backward()

        our_grad_q = q.grad.clone()
        our_grad_k = k.grad.clone()
        our_grad_v = v.grad.clone()

        manager.remove_last_update()
        our_grad_kc2, our_grad_vc2 = manager.grad

        manager.remove_last_update()
        our_grad_kc1, our_grad_vc1 = manager.grad

    print(torch.dist(ref_bwd, our_bwd))
    print(torch.dist(ref_grad_q, our_grad_q))
    print(torch.dist(ref_grad_k, our_grad_k))
    print(torch.dist(ref_grad_v, our_grad_v))
    print(torch.dist(ref_grad_kc1, our_grad_kc1))
    print(torch.dist(ref_grad_vc1, our_grad_vc1))
    print(torch.dist(ref_grad_kc2, our_grad_kc2))
    print(torch.dist(ref_grad_vc2, our_grad_vc2))

    ref_time_bwd.result(detail=True)
    our_time_bwd.result(detail=True)

