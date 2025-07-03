"""
*Experimental* implementation of FlashAttention in Triton.
This file is modified to include a block-sparse version of paged attention.

Original implementation from:
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Key Modifications for Sparse Paged Attention:
- Added `_fwd_kernel_sparse` and `_bwd_kernel_sparse` to handle block-sparse attention.
- Assumes sparsity is defined at the query-block level: all queries in a block attend to the same set of KV blocks.
- The new kernels take `selected_block_indices` and `selected_block_mask` as input.
- Backward pass for sparse attention uses atomic adds for correctness on dK and dV.
- A new user-facing function `flash_paged_sparse_attn_func` is added.
"""

import math

import torch
import triton
import triton.language as tl
from pygments.console import colorize
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
    # Note: It's important to use 'relaxed' semaphore for atomic_add to prevent deadlocks
    # in cases where the same thread block might try to update the same memory location.
    if EVEN_N & EVEN_M:
        tl.atomic_add(dv_ptrs, dv, sem='relaxed')
        tl.atomic_add(dk_ptrs, dk, sem='relaxed')
    else:
        tl.atomic_add(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k, sem='relaxed')
        tl.atomic_add(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k, sem='relaxed')


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

# =================================================================================
# NEW Sparse Paged Attention Kernels
# =================================================================================

@triton.jit
def _fwd_kernel_sparse(
    Q, T,
    selected_block_indices, selected_block_mask,
    Out, Lse,
    TMP,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_ob, stride_oh, stride_om,
    stride_kvh, stride_kvn_page,
    stride_kvbl_b, stride_kvbl_h, stride_kvbl_q,
    nheads, seqlen_q, headdim,
    seqlen_q_rounded, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    INCLUDE_CAUSAL: tl.constexpr
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    # --- 1. Causal attention over the diagonal block ---
    if INCLUDE_CAUSAL:
        causal_block_idx = start_m_block
        
        k_page_ptr = tl.load(T + causal_block_idx * 4)
        v_page_ptr = tl.load(T + causal_block_idx * 4 + 1)
        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))

        offs_n = tl.arange(0, BLOCK_N)
        kv_page_offs = off_kv_h * stride_kvh
        k_ptrs = k_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        v_ptrs = v_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        qk = tl.dot(q, k.T)
        
        # Apply causal mask
        causal_mask = offs_m[:, None] >= (causal_block_idx * BLOCK_N + offs_n)[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_i = tl.maximum(tl.max(qk, 1) * softmax_scale, m_i)
        p = tl.exp(qk * softmax_scale - m_i[:, None])
        lse_i = tl.sum(p, 1)

        acc_o += tl.dot(p.to(v.dtype), v)

    # --- 2. Attention over selected blocks ---
    sel_indices_ptrs = selected_block_indices + off_b * stride_kvbl_b + off_kv_h * stride_kvbl_h + start_m_block * stride_kvbl_q
    sel_mask_ptrs = selected_block_mask + off_b * stride_kvbl_b + off_kv_h * stride_kvbl_h + start_m_block * stride_kvbl_q

    for k_block_idx in range(NUM_SEL_KV_BLOCKS):
        kv_block_idx = tl.load(sel_indices_ptrs + k_block_idx)
        block_mask = tl.load(sel_mask_ptrs + k_block_idx)

        # Do not compute for diagonal blocks in this loop, they are handled in the causal pass
        is_not_diagonal = (kv_block_idx != start_m_block)
        final_mask = block_mask & is_not_diagonal

        k_page_ptr = tl.load(T + kv_block_idx * 4)
        v_page_ptr = tl.load(T + kv_block_idx * 4 + 1)
        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))

        offs_n = tl.arange(0, BLOCK_N)
        kv_page_offs = off_kv_h * stride_kvh
        k_ptrs = k_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        v_ptrs = v_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.dot(q, k.T)
        qk = tl.where(final_mask, qk, float("-inf"))

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


    # --- Finalize and store ---
    o_scale = tl.exp(-lse_i)
    acc_o = acc_o * o_scale[:, None]

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    final_lse = m_i + lse_i
    tl.store(lse_ptrs, final_lse, mask=offs_m < seqlen_q)

    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _bwd_kernel_sparse(
    Q, DO, DQ,
    T, LSE, D,
    selected_block_indices, selected_block_mask,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    stride_kvh, stride_kvn_page,
    stride_kvbl_b, stride_kvbl_h, stride_kvbl_q,
    nheads, seqlen_q, headdim,
    seqlen_q_rounded, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    INCLUDE_CAUSAL: tl.constexpr,
    IS_BF16_ATOM_ADD_SUPPORTED: tl.constexpr
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + (offs_m[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    
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

    # --- 1. Gradients for causal diagonal block ---
    if INCLUDE_CAUSAL:
        causal_block_idx = start_m_block
        k_page_ptr = tl.load(T + causal_block_idx * 4)
        v_page_ptr = tl.load(T + causal_block_idx * 4 + 1)
        dk_page_ptr = tl.load(T + causal_block_idx * 4 + 2)
        dv_page_ptr = tl.load(T + causal_block_idx * 4 + 3)

        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))
        dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))
        dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))

        offs_n = tl.arange(0, BLOCK_N)
        kv_page_offs = off_kv_h * stride_kvh
        k_ptrs = k_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        v_ptrs = v_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        dk_ptrs = dk_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        dv_ptrs = dv_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.dot(q, k.T)
        causal_mask = offs_m[:, None] >= (causal_block_idx * BLOCK_N + offs_n)[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        
        dv_block_causal = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_ptrs, dv_block_causal, sem='relaxed')

        dp = tl.dot(do, v.T)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        dk_block_causal = tl.dot(ds.T, q)
        tl.atomic_add(dk_ptrs, dk_block_causal, sem='relaxed')

        dq_block += tl.dot(ds, k)

    # --- 2. Gradients for selected blocks ---
    sel_indices_ptrs = selected_block_indices + off_b * stride_kvbl_b + off_kv_h * stride_kvbl_h + start_m_block * stride_kvbl_q
    sel_mask_ptrs = selected_block_mask + off_b * stride_kvbl_b + off_kv_h * stride_kvbl_h + start_m_block * stride_kvbl_q

    for k_block_idx in range(NUM_SEL_KV_BLOCKS):
        kv_block_idx = tl.load(sel_indices_ptrs + k_block_idx)
        block_mask = tl.load(sel_mask_ptrs + k_block_idx)
        
        is_not_diagonal = (kv_block_idx != start_m_block)
        final_mask = block_mask & is_not_diagonal
        
        k_page_ptr = tl.load(T + kv_block_idx * 4)
        v_page_ptr = tl.load(T + kv_block_idx * 4 + 1)
        dk_page_ptr = tl.load(T + kv_block_idx * 4 + 2)
        dv_page_ptr = tl.load(T + kv_block_idx * 4 + 3)

        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))
        dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))
        dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))

        offs_n = tl.arange(0, BLOCK_N)
        kv_page_offs = off_kv_h * stride_kvh
        k_ptrs = k_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        v_ptrs = v_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        dk_ptrs = dk_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        dv_ptrs = dv_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.dot(q, k.T)
        qk = tl.where(final_mask, qk, float("-inf"))
        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        dv_block = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_ptrs, dv_block, sem='relaxed')

        dp = tl.dot(do, v.T)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        dk_block = tl.dot(ds.T, q)
        tl.atomic_add(dk_ptrs, dk_block, sem='relaxed')

        dq_block += tl.dot(ds, k)

    # --- Write back dQ ---
    if EVEN_M:
        tl.store(dq_ptrs, dq_block)
    else:
        tl.store(dq_ptrs, dq_block, mask=mask_m[:, None])


def _flash_attn_forward_sparse(
        q: torch.Tensor,
        page_table: torch.Tensor,
        selected_block_indices: torch.Tensor,
        selected_block_mask: torch.Tensor,
        page_size: int,
        num_kv_heads: int,
        kv_head_dim: int,
        causal=False,
        softmax_scale=None):

    batch, seqlen_q, nheads, d = q.shape
    b, h_kv, q_blocks, num_sel = selected_block_indices.shape
    
    assert d <= 128
    assert q.dtype in [torch.float16, torch.bfloat16]
    assert q.is_cuda
    
    BLOCK_M = page_size
    assert seqlen_q % BLOCK_M == 0, "Sequence length must be a multiple of page_size for block-sparse attention"

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    GROUP_SIZE = nheads // num_kv_heads

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    num_warps = 4 if d <= 64 else 8

    stride_kvh = kv_head_dim

    _fwd_kernel_sparse[grid](
        q, page_table,
        selected_block_indices, selected_block_mask,
        o, lse,
        tmp,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        o.stride(0), o.stride(2), o.stride(1),
        stride_kvh, page_size * kv_head_dim, # stride_kvn_page
        selected_block_indices.stride(0), selected_block_indices.stride(1), selected_block_indices.stride(2),
        nheads, seqlen_q, d,
        seqlen_q_rounded, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=page_size,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=(seqlen_q % BLOCK_M == 0),
        NUM_SEL_KV_BLOCKS=num_sel,
        GROUP_SIZE=GROUP_SIZE,
        INCLUDE_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=1
    )

    return o, lse, softmax_scale


def _flash_attn_backward_sparse(
        o: torch.Tensor,
        do: torch.Tensor,
        q: torch.Tensor,
        dq: torch.Tensor,
        page_table: torch.Tensor,
        selected_block_indices: torch.Tensor,
        selected_block_mask: torch.Tensor,
        page_size: int,
        num_kv_heads: int,
        kv_head_dim: int,
        lse: torch.Tensor,
        softmax_scale: float,
        causal: bool = False
):
    if do.stride(-1) != 1:
        do = do.contiguous()

    batch, seqlen_q, nheads, d = q.shape
    b, h_kv, q_blocks, num_sel = selected_block_indices.shape

    BLOCK_M = page_size
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    
    delta = torch.empty_like(lse)
    
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    GROUP_SIZE = nheads // num_kv_heads

    grid_preprocess = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    _bwd_preprocess_do_o_dot[grid_preprocess](
        o, do, delta,
        o.stride(0), o.stride(2), o.stride(1),
        do.stride(0), do.stride(2), do.stride(1),
        nheads, seqlen_q, seqlen_q_rounded, d,
        BLOCK_M=BLOCK_M, BLOCK_HEADDIM=BLOCK_HEADDIM
    )
    
    stride_kvh = kv_head_dim

    grid_bwd = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    _bwd_kernel_sparse[grid_bwd](
        q, do, dq,
        page_table, lse, delta,
        selected_block_indices, selected_block_mask,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        do.stride(0), do.stride(2), do.stride(1),
        dq.stride(0), dq.stride(2), dq.stride(1),
        stride_kvh, page_size * kv_head_dim, # stride_kvn_page
        selected_block_indices.stride(0), selected_block_indices.stride(1), selected_block_indices.stride(2),
        nheads, seqlen_q, d,
        seqlen_q_rounded, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=page_size,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=(seqlen_q % BLOCK_M == 0),
        NUM_SEL_KV_BLOCKS=num_sel,
        GROUP_SIZE=GROUP_SIZE,
        INCLUDE_CAUSAL=causal,
        IS_BF16_ATOM_ADD_SUPPORTED=IS_BF16_ATOM_ADD_SUPPORTED,
        num_warps=4,
        num_stages=1
    )


class FlashPagedSparseAttn(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            selected_block_indices,
            selected_block_mask,
            manager,
            causal):

        q = q if q.stride(-1) == 1 else q.contiguous()

        o, lse, ctx.softmax_scale = _flash_attn_forward_sparse(
            q, manager.page_table,
            selected_block_indices, selected_block_mask,
            manager.page_size, manager.num_kv_heads, manager.head_dim,
            causal=causal, softmax_scale=None)

        ctx.save_for_backward(q, o, lse, selected_block_indices, selected_block_mask)
        ctx.manager = manager
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, o, lse, selected_block_indices, selected_block_mask = ctx.saved_tensors

        with torch.inference_mode():
            dq = torch.zeros_like(q)
            
            _flash_attn_backward_sparse(
                o, do, q, dq,
                ctx.manager.page_table,
                selected_block_indices, selected_block_mask,
                ctx.manager.page_size,
                ctx.manager.num_kv_heads,
                ctx.manager.head_dim,
                lse,
                ctx.softmax_scale,
                causal=ctx.causal
            )

            dk, dv = ctx.manager.grad

        return dq, dk, dv, None, None, None, None


flash_paged_sparse_attn_func = FlashPagedSparseAttn.apply


if __name__ == '__main__':
    from math import ceil
    from functools import partial
    import torch.nn.functional as F
    from einops import einsum, rearrange, reduce, repeat

    # A simple check to ensure 'chunkoptim' is available or provide a mock
    try:
        from chunkoptim.kv_cache import CacheManager
    except ImportError:
        print("警告: 未找到 'chunkoptim'。测试将使用一个模拟的 CacheManager。")
        class CacheManager:
            def __init__(self, batch_size, page_size, num_kv_heads, head_dim):
                self.batch_size = batch_size
                self.page_size = page_size
                self.num_kv_heads = num_kv_heads
                self.head_dim = head_dim
                self.reset()

            def update(self, k, v):
                # This mock assumes k and v have a sequence length divisible by page_size
                k_pages = k.reshape(-1, self.page_size, self.num_kv_heads, self.head_dim)
                v_pages = v.reshape(-1, self.page_size, self.num_kv_heads, self.head_dim)
                
                new_k_pages_ptrs = [p.data_ptr() for p in k_pages]
                new_v_pages_ptrs = [p.data_ptr() for p in v_pages]
                
                num_new_pages = len(new_k_pages_ptrs)
                
                new_dk_pages = torch.zeros_like(k_pages, dtype=torch.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else torch.float32)
                new_dv_pages = torch.zeros_like(v_pages, dtype=torch.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else torch.float32)
                
                new_dk_pages_ptrs = [p.data_ptr() for p in new_dk_pages]
                new_dv_pages_ptrs = [p.data_ptr() for p in new_dv_pages]

                for i in range(num_new_pages):
                    self.page_table_list.extend([
                        new_k_pages_ptrs[i], 
                        new_v_pages_ptrs[i],
                        new_dk_pages_ptrs[i],
                        new_dv_pages_ptrs[i]
                    ])

                self.num_kv += k.shape[1]
                self.all_k_pages.append(k_pages)
                self.all_v_pages.append(v_pages)
                self.all_dk_pages.append(new_dk_pages)
                self.all_dv_pages.append(new_dv_pages)
            
            def reset(self):
                self.page_table_list = []
                self.num_kv = 0
                self.all_k_pages = []
                self.all_v_pages = []
                self.all_dk_pages = []
                self.all_dv_pages = []
            
            @property
            def page_table(self):
                return torch.tensor(self.page_table_list, dtype=torch.int64, device='cuda')
            
            @property
            def grad(self):
                dk_full = torch.cat([p.reshape(-1, self.num_kv_heads, self.head_dim) for p in self.all_dk_pages], dim=0)
                dv_full = torch.cat([p.reshape(-1, self.num_kv_heads, self.head_dim) for p in self.all_dv_pages], dim=0)
                return dk_full.unsqueeze(0), dv_full.unsqueeze(0)


    # A simple check to ensure 'flash_attn' is available or provide a mock
    try:
        from flash_attn import flash_attn_func
        # Use SDPA if flash_attn is not available for masked attention
        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
             flash_attn_func = None
    except ImportError:
        print("警告: 未找到 'flasfrom __future__ import annotations
from native_sparse_attention_pytorch.tensor_typing import Float, Int, Bool

# taken from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
# with fixes for triton 2.3

from functools import partial
import math
from math import ceil

import torch
from torch import Tensor, arange
import torch.nn.functional as F

import einx
from einops import repeat, rearrange, reduce

def exists(v):
    return v is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(n, mult):
    return ceil(n / mult) * mult

def pad_at_dim(t, pad: tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_to_multiple(t, mult, *, dim):
    length = t.shape[dim]
    padded_length = round_up_multiple(length, mult)
    remainder = padded_length - length
    return pad_at_dim(t, (0, remainder), dim = dim)

def is_contiguous(x: Tensor):
    return x.stride(-1) == 1

TRITON_BLOCK_SIZE = 128 # some block size that allows triton not to break, at least half a year ago

INSTALL_COMMAND = 'pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly'

# make sure triton 2.1+ is installed

import packaging.version as pkg_version

import importlib
from importlib.metadata import version

try:
    triton_version = version('triton')
except:
    print(f'latest triton must be installed. `{INSTALL_COMMAND}` first')
    exit()

assert pkg_version.parse(triton_version) >= pkg_version.parse('3.0.0'), f'triton must be version 3.0.0 or above. `{INSTALL_COMMAND}` to upgrade'

import triton
import triton.language as tl
from triton.language.extra import libdevice

# kernels

@triton.jit
def reduce_avg(x, y):
    return (x + y) / 2

@triton.jit
def forward_kernel_causal_and_sparse(
    Q,
    T,
    kv_block_indices,
    kv_block_mask,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    NUM_BLOCKS_PER_SEL: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)

    off_b = off_hb // kv_heads
    off_h = off_hb % kv_heads

    offs_qh = off_h * QUERY_HEAD_GROUPS + tl.arange(0, QUERY_HEAD_GROUPS)

    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q +
        off_b * stride_qb +
        offs_qh[None, :, None] * stride_qh +
        offs_m[:, None, None] * stride_qm +
        offs_d[None, None, :]
    )

    # maximum

    m_i = tl.zeros([BLOCK, QUERY_HEAD_GROUPS], dtype = tl.float32) - float("inf")

    # lse

    lse_ptrs = (
        Lse +
        off_b * stride_lse_b +
        offs_qh[None, :] * seqlen_q_rounded +
        offs_m[:, None]
    )

    lse_i = tl.zeros([BLOCK, QUERY_HEAD_GROUPS], dtype = tl.float32) - float("inf")

    # output

    out_ptrs = (
        Out +
        off_b * stride_ob +
        offs_qh[None, :, None] * stride_oh +
        offs_m[:, None, None] * stride_om +
        offs_d[None, None, :]
    )

    acc_o = tl.zeros([BLOCK,  QUERY_HEAD_GROUPS, BLOCK_HEADDIM], dtype = tl.float32)

    # load queries, keys, values

    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(
                q_ptrs,
                mask = offs_d[None, None, :] < headdim,
                other = 0.0
            )
    else:
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs,
                mask = offs_m[:, None, None] < seqlen_q,
                other = 0.0
            )
        else:
            q = tl.load(
                q_ptrs,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                other = 0.0
            )

    q = q.reshape(BLOCK * QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

    if INCLUDE_BLOCK_CAUSAL:

        # ==================================================================================================
        # NOTE: 新增加
        k_page_ptrs = tl.load(T + (tl.arange(NUM_BLOCKS_PER_SEL) + start_m - NUM_BLOCKS_PER_SEL + 1) * 4)
        v_page_ptrs = tl.load(T + (tl.arange(NUM_BLOCKS_PER_SEL) + start_m - NUM_BLOCKS_PER_SEL + 1) * 4 + 1)
        k_page_ptrs = tl.expand_dims(k_page_ptrs, axis=1)
        v_page_ptrs = tl.expand_dims(v_page_ptrs, axis=1)
        inner_page_offs = tl.expand_dims(tl.arange(BLOCK), axis=0)
        # ==================================================================================================

        start_offs_n = (
            start_m * BLOCK +
            tl.arange(0, SEL_BLOCK) - (SEL_BLOCK - BLOCK)
        )

        if SLIDING:
            tl.device_print("error sliding", 0)
            num_kv_blocks = 2
        else:
            num_kv_blocks = 1

        for kv_block_offset_ind in range(num_kv_blocks):
            offset = kv_block_offset_ind * -SEL_BLOCK

            offs_n = start_offs_n + offset

            # ==================================================================
            # NOTE: 修改
            k_ptrs = (
                # K +
                k_page_ptrs[:, None, None], # (num_blks_per_sel, 1, 1)
                off_b * stride_kb +
                off_h * stride_kh +
                # offs_n[:, None] * stride_kn +
                inner_page_offs[None, :, None] * stride_kn + # (1, block-size, 1)
                # offs_d[None, :]
                offs_d[None, None, :]
            ).reshape(NUM_BLOCKS_PER_SEL * BLOCK, BLOCK_HEADDIM)
            # ==================================================================


            # ==================================================
            # NOTE: 修改
            v_ptrs = (
                # V +
                v_page_ptrs[:, None, None],
                off_b * stride_vb +
                off_h * stride_vh +
                # offs_n[:, None] * stride_vn +
                inner_page_offs[None, :, None] * stride_vn +
                # offs_d[None, :]
                offs_d[None, None, :]
            ).reshape(NUM_BLOCKS_PER_SEL * BLOCK, BLOCK_HEADDIM)
            # ==================================================


            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    k = tl.load(
                        k_ptrs,
                        mask = (offs_n[:, None] >= 0),
                        other = 0.
                    )
                else:
                    k = tl.load(
                        k_ptrs,
                        mask = (offs_n[:, None] >= 0) & (offs_d[None, :] < headdim),
                        other = 0.0
                    )
            else:
                if EVEN_HEADDIM:
                    k = tl.load(
                        k_ptrs,
                        mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seqlen_k),
                        other = 0.0,
                    )
                else:
                    k = tl.load(
                        k_ptrs,
                        mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                        other = 0.0,
                    )

            qk = tl.zeros([BLOCK * QUERY_HEAD_GROUPS, SEL_BLOCK], dtype=tl.float32)

            qk += tl.dot(q, tl.trans(k))

            qk = qk.reshape(BLOCK, QUERY_HEAD_GROUPS, SEL_BLOCK)

            if BLOCK != SEL_BLOCK and not SLIDING:
                block_diagonal_mask = (
                    (offs_n[None, None, :] >= 0.) &
                    ((offs_n[None, None, :] // SEL_BLOCK) == (offs_m[:, None, None] // SEL_BLOCK))
                )

                qk += tl.where(block_diagonal_mask, 0, float('-inf'))

            if not EVEN_N:
                qk += tl.where(offs_n[None, :] < seqlen_k, 0, float('-inf'))

            qk = qk.reshape(BLOCK, QUERY_HEAD_GROUPS, SEL_BLOCK)

            causal_mask = offs_m[:, None, None] >= offs_n[None, None, :]

            if SLIDING:
                causal_mask &= (
                    (offs_n[None, None, :] >= 0.) &
                    ((offs_m[:, None, None] - offs_n[None, None, :]) <= SEL_BLOCK)
                )

            qk += tl.where(causal_mask, 0, float("-inf"))

            m_ij = tl.maximum(tl.max(qk, 2) * softmax_scale, m_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, :, None])

            l_ij = tl.sum(p, 2)

            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o *= acc_o_scale[:, :, None]

            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    v = tl.load(
                        v_ptrs,
                        mask = (offs_n[:, None] >= 0),
                        other = 0.
                    )
                else:
                    v = tl.load(
                        v_ptrs,
                        mask = (offs_n[:, None] >= 0) & (offs_d[None, :] < headdim),
                        other = 0.0
                    )
            else:
                if EVEN_HEADDIM:
                    v = tl.load(
                        v_ptrs,
                        mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seqlen_k),
                        other = 0.0,
                    )
                else:
                    v = tl.load(
                        v_ptrs,
                        mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                        other = 0.0,
                    )

            p = p.reshape(BLOCK * QUERY_HEAD_GROUPS, SEL_BLOCK).to(v.dtype)

            causal_o = tl.dot(p, v)

            acc_o += causal_o.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

            # -- update statistics

            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    # # take care of the selected kv blocks

    # (BLOCK,)
    kv_block_indices_ptrs = (
        kv_block_indices +
        off_b * stride_kvbl_b +
        off_h * stride_kvbl_h +
        offs_m * stride_kvbl_m
    )

    # (BLOCK,)
    kv_block_mask_ptrs = (
        kv_block_mask +
        off_b * stride_kvbl_b +
        off_h * stride_kvbl_h +
        offs_m * stride_kvbl_m
    )

    q = q.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)
    q = tl.expand_dims(q, 2)
    q = tl.broadcast_to(q, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM))
    q = q.reshape(BLOCK, 16, BLOCK_HEADDIM)

    for off_sel_kv_block in range(NUM_SEL_KV_BLOCKS):

        # (BLOCK,)
        block_indices = tl.load(
            kv_block_indices_ptrs + off_sel_kv_block,
            mask = offs_m < seqlen_q,
            other = 0
        )

        # (BLOCK,)
        block_masks = tl.load(
            kv_block_mask_ptrs + off_sel_kv_block,
            mask = offs_m < seqlen_q,
            other = False
        )

        for off_blocks_per_sel in range(NUM_BLOCKS_PER_SEL):

            # =========================================================================================
            # NOTE: 新增加
            k_page_ptrs = tl.load(T + (block_indices * NUM_BLOCKS_PER_SEL + off_blocks_per_sel) * 4)
            v_page_ptrs = tl.load(T + (block_indices * NUM_BLOCKS_PER_SEL + off_blocks_per_sel) * 4 + 1)
            inner_page_offs = tl.arange(0, BLOCK)
            # =========================================================================================

            # (BLOCK, BLOCK)
            blocks_offs_n = (
                block_indices[:, None] * (BLOCK * NUM_BLOCKS_PER_SEL) +
                tl.arange(0, BLOCK)[None, :] + (off_blocks_per_sel * BLOCK)
            )

            # ==============================================
            # NOTE: 修改
            # (BLOCK, BLOCK, HEAD_DIM), 
            # 分别表示 m 个sel blocks的第一个小block
            block_k_ptrs = (
                # K +
                k_page_ptrs[:, None, None] +
                off_b * stride_kb +
                off_h * stride_kh +
                # blocks_offs_n[:, :, None] * stride_kn +
                inner_page_offs[None, :, None] * stride_kn,
                offs_d[None, None, :]
            )

            block_v_ptrs = (
                # V +
                v_page_ptrs[:, None, None] + 
                off_b * stride_vb +
                off_h * stride_vh + 
                # blocks_offs_n[:, :, None] * stride_vn +
                inner_page_offs[None, :, None] * stride_vn +
                offs_d[None, None, :]
            )
            # ==============================================

            # load k of shape (m, n, d), sparsely selected by each query

            # ==============================================
            # NOTE: 修改
            k_block = tl.load(
                block_k_ptrs,
                mask = blocks_offs_n[:, :, None] < seqlen_k,
                other = 0.
            )
            # ==============================================

            # similarities

            block_qk = tl.zeros([BLOCK, 16, BLOCK], dtype = tl.float32)
            sel_qk = tl.zeros([BLOCK, QUERY_HEAD_GROUPS, BLOCK], dtype = tl.float32)

            k_block = k_block.reshape(BLOCK, BLOCK, BLOCK_HEADDIM)
            k_block = k_block.permute(0, 2, 1)

            block_qk += tl.dot(q, k_block)
            block_qk = block_qk.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK)
            block_qk = tl.reduce(block_qk, 2, reduce_avg)

            sel_qk += block_qk
            sel_qk += tl.where(block_masks[:, None, None], 0, float("-inf"))

            # attention

            m_ij = tl.maximum(tl.max(sel_qk, 2) * softmax_scale, m_i)
            block_p = tl.exp(sel_qk * softmax_scale - m_ij[:, :, None])

            l_ij = tl.sum(block_p, 2)

            # renormalize the running output

            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, :, None]

            # aggregate values

            # ==============================================
            # NOTE: 修改
            v_block = tl.load(
                block_v_ptrs,
                mask = blocks_offs_n[:, :, None] < seqlen_k,
                other = 0.
            )
            # ==============================================

            v_block = tl.reshape(v_block, (BLOCK, BLOCK, BLOCK_HEADDIM))

            block_p = block_p.to(v_block.dtype)
            p_expanded = block_p.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK)
            p_expanded = tl.expand_dims(p_expanded, 2)
            p_expanded = tl.broadcast_to(p_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK))
            p_expanded = p_expanded.reshape(BLOCK, 16, BLOCK)

            block_acc_o = tl.dot(p_expanded, v_block)
            block_acc_o = block_acc_o.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM)
            block_acc_o = tl.reduce(block_acc_o, 2, reduce_avg)

            acc_o += block_acc_o

            # -- update statistics

            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    # normalize accumulated out

    acc_o_scale = tl.exp(m_i - lse_i)
    acc_o *= acc_o_scale[:, :, None]

    # write back lse

    lse_i = lse_i.reshape(BLOCK, QUERY_HEAD_GROUPS)
    tl.store(lse_ptrs, lse_i)

    # write to output

    acc_o = acc_o.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask = offs_d[None, None, :] < headdim
            )
    else:
        if EVEN_HEADDIM:
            tl.store(
                out_ptrs,
                acc_o,
                mask = offs_m[:, None, None] < seqlen_q
            )
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim)
            )

@triton.heuristics(
    dict(
        EVEN_M = lambda args: divisible_by(args["seqlen_q"], args["BLOCK"]),
        EVEN_N = lambda args: divisible_by(args["seqlen_k"], args["BLOCK"]),
        EVEN_HEADDIM = lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        QUERY_EXPAND_DIM = lambda args: 16 // args['QUERY_HEAD_GROUPS']
    )
)
@triton.jit
def forward_kernel(
    Q,
    T,
    kv_block_indices,
    kv_block_mask,
    Out,
    SlidingOut,
    Lse,
    SlidingLse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    NUM_BLOCKS_PER_SEL: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    RETURN_SLIDING_OUT: tl.constexpr
):
    if RETURN_SLIDING_OUT:
        sliding = tl.program_id(2) == 0
        out_ptr = SlidingOut if sliding else Out
        lse_ptr = SlidingLse if sliding else Lse
        num_sel_kv_blocks = 0 if sliding else NUM_SEL_KV_BLOCKS
    else:
        sliding = False
        out_ptr = Out
        lse_ptr = Lse
        num_sel_kv_blocks = NUM_SEL_KV_BLOCKS

    forward_kernel_causal_and_sparse(
        Q,
        T,
        kv_block_indices,
        kv_block_mask,
        out_ptr,
        lse_ptr,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        stride_om,
        stride_kvbl_b,
        stride_kvbl_h,
        stride_kvbl_m,
        stride_lse_b,
        kv_heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        BLOCK_HEADDIM,
        EVEN_M,
        EVEN_N,
        EVEN_HEADDIM,
        BLOCK,
        SEL_BLOCK,
        QUERY_HEAD_GROUPS,
        QUERY_EXPAND_DIM,
        num_sel_kv_blocks,
        NUM_BLOCKS_PER_SEL,
        INCLUDE_BLOCK_CAUSAL,
        sliding
    )

def native_sparse_attn_forward(
    q,
    page_table,
    kv_block_indices,
    kv_block_mask,
    kv_heads,
    seqlen_k,
    block_size = 128,
    include_block_causal = True,
    return_sliding_window_out = False,
):
    # q, k, v, kv_block_indices = [x if is_contiguous(x) else x.contiguous() for x in (q, k, v, kv_block_indices)]
    q, page_table, kv_block_indices = [x if is_contiguous(x) else x.contiguous() for x in (q, page_table, kv_block_indices)]

    batch, nheads, seqlen_q, dim, device = *q.shape, q.device
    # _, kv_heads, seqlen_k, _ = k.shape
    assert divisible_by(nheads, kv_heads)
    head_groups = nheads // kv_heads

    assert divisible_by(block_size, 16)

    num_blocks_per_sel = block_size // 16
    num_selected_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape

    # assert k.shape == (batch, kv_heads, seqlen_k, dim)
    # assert v.shape == (batch, kv_heads, seqlen_k, dim)
    assert dim <= 128, "only support head dimensions up to 128"
    # assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert all([t.is_cuda for t in (q, page_table)])

    softmax_scale = dim ** -0.5

    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)

    lse = torch.empty((batch, nheads, seqlen_q_rounded), device = device, dtype = torch.float32)
    slide_lse = torch.empty((batch, nheads, seqlen_q_rounded), device = device, dtype = torch.float32)

    o = torch.empty_like(q)
    slide_o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)
    num_warps = 4 if dim <= 64 else 8

    grid = lambda META: (
        triton.cdiv(seqlen_q, META["BLOCK"]),
        batch * kv_heads,
        (2 if return_sliding_window_out else 1)
    ) # kv heads here, as grouped query heads all loaded, following the paper

    # ===============================================================
    # NOTE: 新增加
    stride_kb = kv_heads * 16 * dim
    stride_kh = 16 * dim
    stride_kn = dim
    stride_vb, stride_vh, stride_vn = stride_kb, stride_kh, stride_kn
    # ===============================================================

    forward_kernel[grid](
        q,
        page_table,
        kv_block_indices,
        kv_block_mask,
        o,
        slide_o,
        lse,
        slide_lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        # k.stride(0),
        # k.stride(1),
        # k.stride(2),
        # v.stride(0),
        # v.stride(1),
        # v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_block_indices.stride(0),
        kv_block_indices.stride(1),
        kv_block_indices.stride(2),
        lse.stride(0),
        kv_heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        dim,
        seqlen_q // 32,
        seqlen_k // 32,
        BLOCK_HEADDIM,
        BLOCK = 16,
        SEL_BLOCK = block_size,
        QUERY_HEAD_GROUPS = head_groups,
        NUM_SEL_KV_BLOCKS = num_selected_fine_blocks,
        NUM_BLOCKS_PER_SEL = num_blocks_per_sel,
        INCLUDE_BLOCK_CAUSAL = include_block_causal,
        RETURN_SLIDING_OUT = return_sliding_window_out,
        num_warps = num_warps,
        num_stages = 1,
    )

    return o, slide_o, lse, slide_lse

@triton.jit
def backward_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    qheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // qheads
    off_h = off_hb % qheads

    # initialize offsets

    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # load

    o = tl.load(
        Out +
        off_b * stride_ob +
        off_h * stride_oh +
        offs_m[:, None] * stride_om +
        offs_d[None, :],
        mask = (
            (offs_m[:, None] < seqlen_q) &
            (offs_d[None, :] < headdim)
        ),
        other = 0.0,
    ).to(tl.float32)

    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask = (
            offs_m[:, None] < seqlen_q) &
            (offs_d[None, :] < headdim
        ),
        other = 0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    # write-back

    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

@triton.jit
def backward_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.atomic_add(dv_ptrs, dv, sem = 'relaxed')
            tl.atomic_add(dk_ptrs, dk, sem = 'relaxed')
        else:
            tl.atomic_add(dv_ptrs, dv, mask=offs_d[None, :] < headdim, sem = 'relaxed')
            tl.atomic_add(dk_ptrs, dk, mask=offs_d[None, :] < headdim, sem = 'relaxed')
    else:
        if EVEN_HEADDIM:
            tl.atomic_add(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k, sem = 'relaxed')
            tl.atomic_add(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k, sem = 'relaxed')
        else:
            tl.atomic_add(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), sem = 'relaxed')
            tl.atomic_add(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), sem = 'relaxed')


@triton.jit
def backward_kernel_one_col_block_sparse(
    start_n,
    Q,
    T,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    DO,
    DQ,
    LSE,
    D,
    softmax_scale,
    kv_base_off,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    stride_kvbl_m,
    stride_qh,
    stride_doh,
    stride_dqh,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    RETURN_SEL_GRADS: tl.constexpr,
    OFF_SEL_KV_BLOCKS: tl.constexpr,
    NUM_BLOCKS_PER_SEL: tl.constexpr,
    OFF_BLOCK_PER_SEL: tl.constexpr,
    BLOCK_DV_USE_DOT: tl.constexpr,
    BLOCK_DK_USE_DOT: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)

    begin_m = ((start_n * BLOCK) // BLOCK) * BLOCK

    # initialize row/col offsets

    offs_qm = begin_m + tl.arange(0, BLOCK)
    offs_n = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_m = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    offs_g = tl.arange(0, QUERY_HEAD_GROUPS)

    offs_d_or_lse = seqlen_q_rounded * offs_g[:, None] + offs_m

    # initialize pointers to value-like data

    q_ptrs = (
        Q +
        offs_g[None, :, None] * stride_qh +
        offs_qm[:, None, None] * stride_qm +
        offs_d[None, None, :]
    )

    do_ptrs = (
        DO +
        offs_g[None, :, None] * stride_doh +
        offs_qm[:, None, None] * stride_dom +
        offs_d[None, None, :]
    )

    dq_ptrs = (
        DQ +
        offs_g[None, :, None] * stride_dqh +
        offs_qm[:, None, None] * stride_dqm +
        offs_d[None, None, :]
    )

    # same block for block causal diagonal

    # load q, k, v, do on-chip
    # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs,
                mask = offs_m[:, None, None] < seqlen_q,
                other = 0.0
            )
        else:
            q = tl.load(
                q_ptrs,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                other = 0.0,
            )

    # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
    # Also wrong for headdim=64.

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    lse_i = tl.load(LSE + offs_d_or_lse)
    lse_i = tl.trans(lse_i) # (m, h)

    # compute dv
    # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
    # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
    # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
    # the output is correct.
    if EVEN_M & EVEN_HEADDIM:
        do = tl.load(do_ptrs)
    else:
        # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
        do = tl.load(
            do_ptrs,
            mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
            other = 0.0,
        )

    # compute dp = dot(v, do)
    # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
    # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
    # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    # There's a race condition for headdim=48
    if not EVEN_HEADDIM:
        tl.debug_barrier()

    # compute ds = p * (dp - delta[:, None])
    # Putting the subtraction after the dp matmul (instead of before) is slightly faster

    Di = tl.load(D + offs_d_or_lse)
    Di = tl.trans(Di) # (m, h)

    # Converting ds to q.dtype here reduces register pressure and makes it much faster
    # for BLOCK_HEADDIM=128

    dq = tl.zeros([BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM], dtype = tl.float32)

    # handle kv block indices using atomic adds for starters, todo: swap dq and dk/dv loops at some point, semi big refactor

    # (BLOCK,)
    kv_block_indices_ptrs = (
        kv_block_indices +
        offs_m * stride_kvbl_m
    )

    kv_block_mask_ptrs = (
        kv_block_mask +
        offs_m * stride_kvbl_m
    )

    # (BLOCK,)
    block_indices = tl.load(
        kv_block_indices_ptrs + OFF_SEL_KV_BLOCKS,
        mask = offs_m < seqlen_q,
        other = 0.
    )

    block_masks = tl.load(
        kv_block_mask_ptrs + OFF_SEL_KV_BLOCKS,
        mask = offs_m < seqlen_q,
        other = 0.
    )

    # (BLOCK,BLOCK)
    blocks_offs_n = (
        block_indices[:, None] * (BLOCK * NUM_BLOCKS_PER_SEL) +
        tl.arange(0, BLOCK)[None, :] + (OFF_BLOCK_PER_SEL * BLOCK)
    )


    # ========================================================================================================
    # NOTE: 新增加
    k_page_ptrs = tl.load(T + (block_indices * NUM_BLOCKS_PER_SEL + OFF_BLOCK_PER_SEL) * 4) + kv_base_off
    v_page_ptrs = tl.load(T + (block_indices * NUM_BLOCKS_PER_SEL + OFF_BLOCK_PER_SEL) * 4 + 1) + kv_base_off
    dk_page_ptrs = tl.load(T + (block_indices * NUM_BLOCKS_PER_SEL + OFF_BLOCK_PER_SEL) * 4 + 2) + kv_base_off
    dv_page_ptrs = tl.load(T + (block_indices * NUM_BLOCKS_PER_SEL + OFF_BLOCK_PER_SEL) * 4 + 3) + kv_base_off
    inner_page_offs = tl.arange(0, BLOCK)
    # ========================================================================================================


    # ===============================================
    # NOTE: 修改
    block_k_ptrs = (
        # K + 
        k_page_ptrs[:, None, None] +
        # blocks_offs_n[:, :, None] * stride_kn +
        inner_page_offs[None, :, None] * stride_kn +
        offs_d[None, None, :]
    )

    block_v_ptrs = (
        # V + 
        v_page_ptrs[:, None, None] +
        # blocks_offs_n[:, :, None] * stride_vn +
        inner_page_offs[None, :, None] * stride_vn,
        offs_d[None, None, :]
    )

    block_dv_ptrs = (
        # DV +
        dv_page_ptrs[:, None, None],
        # blocks_offs_n[:, :, None] * stride_dvn +
        inner_page_offs[None, :, None] * stride_dvn,
        offs_d[None, None, :]
    )

    block_dk_ptrs = (
        # DK +
        dk_page_ptrs[:, None, None],
        # blocks_offs_n[:, :, None] * stride_dkn +
        inner_page_offs[None, :, None] * stride_dkn,
        offs_d[None, None, :]
    )
    # ===============================================

    block_k = tl.load(
        block_k_ptrs,
        mask = blocks_offs_n[:, :, None] < seqlen_k,
        other = 0.
    )

    block_v = tl.load(
        block_v_ptrs,
        mask = blocks_offs_n[:, :, None] < seqlen_k,
        other = 0.
    )

    q_expanded = tl.expand_dims(q, 2)
    q_expanded = tl.broadcast_to(q_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM))
    q_expanded = q_expanded.reshape(BLOCK, 16, BLOCK_HEADDIM)

    block_k_permuted = tl.permute(block_k, (0, 2, 1))
    block_qk = tl.dot(q_expanded, block_k_permuted)

    block_qk = block_qk.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK)
    qk = tl.reduce(block_qk, 2, reduce_avg)

    masked_qk = qk + tl.where(block_masks[:, None, None], 0, float("-inf"))

    p = tl.exp(masked_qk * softmax_scale - lse_i[:, :, None])

    # prepare do

    do_expanded = tl.expand_dims(do, 2)
    do_expanded = tl.broadcast_to(do_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM))
    do_expanded = do_expanded.reshape(BLOCK, 16, BLOCK_HEADDIM)

    # take care of block dv

    if BLOCK_DV_USE_DOT:
        p_expanded = p.to(do.dtype)
        p_expanded = tl.expand_dims(p_expanded, 2)
        p_expanded = tl.broadcast_to(p_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK))
        p_expanded = p_expanded.reshape(BLOCK, QUERY_HEAD_GROUPS * QUERY_EXPAND_DIM, BLOCK)
        p_expanded = tl.permute(p_expanded, (0, 2, 1))

        block_dv = tl.dot(p_expanded, do_expanded) / QUERY_EXPAND_DIM
    else:
        block_dv = p.to(do.dtype)[:, :, :, None] * do[:, :, None, :]
        block_dv = tl.sum(block_dv, 1)

    tl.atomic_add(
        block_dv_ptrs, block_dv,
        mask = block_masks[:, None, None] & blocks_offs_n[:, :, None] < seqlen_k,
        sem = 'relaxed'
    )

    # get dp

    block_v = tl.permute(block_v, (0, 2, 1))

    dp = tl.dot(do_expanded, block_v)

    dp = dp.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK)
    dp = tl.reduce(dp, 2, reduce_avg)

    # ds

    ds = (p * (dp - Di[:, :, None]) * softmax_scale)

    # maybe return gradients for better differentiable topk

    if RETURN_SEL_GRADS:

        kv_block_grads_ptrs = (
            kv_block_grads +
            offs_m * stride_kvbl_m
        )

        sel_grads = ds * qk # (q block, q head group, k block)

        sel_grads = tl.where(block_masks[:, None, None], sel_grads, 0.)

        sel_grads = tl.sum(sel_grads, 2) # for k block
        sel_grads = tl.sum(sel_grads, 1) # for q head groups

        tl.atomic_add(
            kv_block_grads_ptrs + OFF_SEL_KV_BLOCKS,
            sel_grads,
            mask = (offs_m < seqlen_q),
            sem = 'relaxed'
        )

    # ds

    ds_expanded = tl.expand_dims(ds, 2)
    ds_expanded = tl.broadcast_to(ds_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK))
    ds_expanded = ds_expanded.reshape(BLOCK, 16, BLOCK)

    # block dk

    if BLOCK_DK_USE_DOT:
        ds_permuted = tl.permute(ds_expanded, (0, 2, 1))
        block_dk = tl.dot(ds_permuted.to(q_expanded.dtype), q_expanded) / QUERY_EXPAND_DIM
    else:
        block_dk = ds[:, :, :, None] * q[:, :, None, :].to(ds.dtype)
        block_dk = tl.sum(block_dk, 1)

    tl.atomic_add(
        block_dk_ptrs,
        block_dk,
        mask = block_masks[:, None, None] & blocks_offs_n[:, :, None] < seqlen_k,
        sem = 'relaxed'
    )

    # block dq

    block_dq = tl.dot(ds_expanded.to(block_k.dtype), block_k)

    block_dq = block_dq.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM)
    block_dq = tl.reduce(block_dq, 2, reduce_avg)

    dq += block_dq

    # update dq

    dq = dq.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

    if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
        tl.atomic_add(dq_ptrs, dq, sem = 'relaxed')
    else:
        if EVEN_HEADDIM:
            tl.atomic_add(
                dq_ptrs,
                dq,
                mask = offs_m[:, None, None] < seqlen_q,
                sem = 'relaxed'
            )
        else:
            tl.atomic_add(
                dq_ptrs,
                dq,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                sem = 'relaxed',
            )

@triton.jit
def backward_kernel_one_col_block_causal(
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
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    stride_kvbl_m,
    stride_qh,
    stride_doh,
    stride_dqh,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    SLIDING: tl.constexpr
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)

    begin_m = ((start_n * BLOCK) // BLOCK) * BLOCK

    # initialize row/col offsets

    offs_qm = begin_m + tl.arange(0, BLOCK)
    offs_n = start_n * BLOCK + tl.arange(0, SEL_BLOCK) - (SEL_BLOCK - BLOCK)
    offs_m = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    offs_g = tl.arange(0, QUERY_HEAD_GROUPS)

    offs_d_or_lse = seqlen_q_rounded * offs_g[:, None] + offs_m

    # initialize pointers to value-like data

    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])

    q_ptrs = (
        Q +
        offs_g[:, None, None] * stride_qh +
        offs_qm[None, :, None] * stride_qm +
        offs_d[None, None, :]
    )

    do_ptrs = (
        DO +
        offs_g[:, None, None] * stride_doh +
        offs_qm[None, :, None] * stride_dom +
        offs_d[None, None, :]
    )

    dq_ptrs = (
        DQ +
        offs_g[:, None, None] * stride_dqh +
        offs_qm[None, :, None] * stride_dqm +
        offs_d[None, None, :]
    )

    # initialize dv and dk

    dv = tl.zeros([SEL_BLOCK, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([SEL_BLOCK, BLOCK_HEADDIM], dtype=tl.float32)

    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs,
                mask = (offs_n[:, None] >= 0),
                other = 0.
            )
            v = tl.load(
                v_ptrs,
                mask = (offs_n[:, None] >= 0),
                other = 0.
            )
        else:
            k = tl.load(
                k_ptrs,
                mask = (offs_n[:, None] >= 0) & (offs_d[None, :] < headdim),
                other = 0.0
            )

            v = tl.load(
                v_ptrs,
                mask = (offs_n[:, None] >= 0) & (offs_d[None, :] < headdim),
                other = 0.0
            )
    else:
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs,
                mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seqlen_k),
                other = 0.0
            )

            v = tl.load(
                v_ptrs,
                mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seqlen_k),
                other = 0.0
            )
        else:
            k = tl.load(
                k_ptrs,
                mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other = 0.0
            )

            v = tl.load(
                v_ptrs,
                mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other = 0.0
            )

    # same block for block causal diagonal

    # load q, k, v, do on-chip
    # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[None, :, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[None, :, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                other=0.0,
            )
    # recompute p = softmax(qk, dim=-1).T

    q = q.reshape([QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM])

    qk = tl.dot(q, tl.trans(k))

    qk = qk.reshape(QUERY_HEAD_GROUPS, BLOCK, SEL_BLOCK)

    mask = offs_m[:, None] >= offs_n[None, :]

    if BLOCK != SEL_BLOCK and not SLIDING:
        block_diagonal_mask = (
            (offs_n[None, :] >= 0) &
            ((offs_n[None, :] // SEL_BLOCK) == (offs_m[:, None] // SEL_BLOCK))
        )

        mask &= block_diagonal_mask

    # Trying to combine the two masks seem to make the result wrong
    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        mask &= offs_n[None, :] < seqlen_k

    if SLIDING:
        mask &= (
            (offs_n[None, :] >= 0.) &
            (offs_m[:, None] - offs_n[None, :]) <= SEL_BLOCK
        )

    qk = tl.where(mask, qk, float("-inf"))

    qk = qk.reshape(QUERY_HEAD_GROUPS * BLOCK, SEL_BLOCK)

    # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
    # Also wrong for headdim=64.

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    lse_i = tl.load(LSE + offs_d_or_lse)
    lse_i = lse_i.reshape(QUERY_HEAD_GROUPS * BLOCK)

    p = tl.exp(qk * softmax_scale - lse_i[:, None])

    # compute dv
    # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
    # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
    # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
    # the output is correct.
    if EVEN_M & EVEN_HEADDIM:
        do = tl.load(do_ptrs)
    else:
        # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
        do = tl.load(
            do_ptrs,
            mask = (offs_m[None, :, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
            other = 0.0,
        )

    do = do.reshape(QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM)

    dv += tl.dot(tl.trans(p.to(do.dtype)), do)

    # compute dp = dot(v, do)
    # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
    # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
    # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    dp = tl.dot(do, tl.trans(v))

    # There's a race condition for headdim=48
    if not EVEN_HEADDIM:
        tl.debug_barrier()

    # compute ds = p * (dp - delta[:, None])
    # Putting the subtraction after the dp matmul (instead of before) is slightly faster

    Di = tl.load(D + offs_d_or_lse)
    Di = Di.reshape(QUERY_HEAD_GROUPS * BLOCK)

    # Converting ds to q.dtype here reduces register pressure and makes it much faster
    # for BLOCK_HEADDIM=128

    ds = (p * (dp - Di[:, None]) * softmax_scale)

    ds = ds.to(q.dtype)

    # compute dk = dot(ds.T, q)

    dk += tl.dot(tl.trans(ds), q)

    # compute dq

    if not (
        EVEN_M & EVEN_HEADDIM
    ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
        tl.debug_barrier()

    dq = tl.zeros([QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM], dtype = tl.float32)

    dq += tl.dot(ds, k)

    # update dq

    dq = dq.reshape(QUERY_HEAD_GROUPS, BLOCK, BLOCK_HEADDIM)

    if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
        tl.atomic_add(dq_ptrs, dq, sem = 'relaxed')
    else:
        if EVEN_HEADDIM:
            tl.atomic_add(dq_ptrs, dq, mask=offs_m[None, :, None] < seqlen_q, sem = 'relaxed')
        else:
            tl.atomic_add(
                dq_ptrs,
                dq,
                mask = (offs_m[None, :, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                sem = 'relaxed',
            )

    # write-back

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])

    backward_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M = EVEN_M,
        EVEN_N = EVEN_N,
        EVEN_HEADDIM = EVEN_HEADDIM,
    )

@triton.heuristics(
    dict(
        QUERY_EXPAND_DIM = lambda args: 16 // args['QUERY_HEAD_GROUPS']
    )
)
@triton.jit
def backward_kernel(
    Q,
    T,
    # K,
    # V,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    DO,
    DQ,
    # DK,
    # DV,
    LSE,
    D,
    SLIDE_DO,
    SLIDE_LSE,
    SLIDE_D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    stride_D_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    RETURN_SEL_GRADS: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr,
    NUM_BLOCKS_PER_SEL: tl.constexpr,
    BLOCK_DV_USE_DOT: tl.constexpr,
    BLOCK_DK_USE_DOT: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // kv_heads
    off_h = off_hb % kv_heads
    off_qh = off_h * QUERY_HEAD_GROUPS

    # determine whether block causal diagonal, sliding, or selected fine kv blocks

    block_id = tl.program_id(0)

    IS_CAUSAL = False
    IS_SLIDING = False

    do = DO
    lse = LSE
    delta = D

    if INCLUDE_BLOCK_CAUSAL:
        IS_CAUSAL = block_id == 0
        block_id -= 1

    if SLIDING:
        IS_SLIDING = block_id == 0
        block_id -= 1

    if IS_SLIDING:
        do = SLIDE_DO
        lse = SLIDE_LSE
        delta = SLIDE_D

    OFF_SEL_KV_BLOCKS = block_id // NUM_BLOCKS_PER_SEL
    OFF_BLOCK_PER_SEL = block_id % NUM_BLOCKS_PER_SEL

    # offset pointers for batch/head

    Q += off_b * stride_qb + off_qh * stride_qh

    # ================================================
    # NOTE: 修改
    # K += off_b * stride_kb + off_h * stride_kh
    # V += off_b * stride_vb + off_h * stride_vh
    kv_base_off = off_b * stride_kb + off_h * stride_kh
    # ================================================

    DQ += off_b * stride_dqb + off_qh * stride_dqh

    # ================================================
    # NOTE: 注释掉
    # DK += off_b * stride_dkb + off_h * stride_dkh
    # DV += off_b * stride_dvb + off_h * stride_dvh
    # ================================================

    do += off_b * stride_dob + off_qh * stride_doh

    # offset pointers for batch/head for selected kv block related

    kv_block_indices += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    kv_block_mask += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    kv_block_grads += off_b * stride_kvbl_b + off_h * stride_kvbl_h

    # pointer to row-wise quantities in value-like data

    delta += (
        off_b * stride_D_b +
        off_qh * seqlen_q_rounded
    )

    lse += (
        off_b * stride_lse_b +
        off_qh * seqlen_q_rounded
    )

    start_n = tl.program_id(2)

    if IS_CAUSAL or IS_SLIDING:
        tl.device_print("error IS_CAUSAL", 0)
        backward_kernel_one_col_block_causal(
            start_n,
            Q,
            T,
            do,
            DQ,
            lse,
            delta,
            softmax_scale,
            kv_base_off,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            stride_kvbl_m,
            stride_qh,
            stride_doh,
            stride_dqh,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            headdim,
            BLOCK_HEADDIM = BLOCK_HEADDIM,
            EVEN_M = EVEN_M,
            EVEN_N = EVEN_N,
            EVEN_HEADDIM = EVEN_HEADDIM,
            BLOCK = BLOCK,
            SEL_BLOCK = SEL_BLOCK,
            QUERY_HEAD_GROUPS = QUERY_HEAD_GROUPS,
            QUERY_EXPAND_DIM = QUERY_EXPAND_DIM,
            SLIDING = IS_SLIDING
        )
    else:
        backward_kernel_one_col_block_sparse(
            start_n,
            Q,
            T,
            kv_block_indices,
            kv_block_mask,
            kv_block_grads,
            do,
            DQ,
            lse,
            delta,
            softmax_scale,
            kv_base_off,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            stride_kvbl_m,
            stride_qh,
            stride_doh,
            stride_dqh,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            headdim,
            BLOCK_HEADDIM = BLOCK_HEADDIM,
            EVEN_M = EVEN_M,
            EVEN_N = EVEN_N,
            EVEN_HEADDIM = EVEN_HEADDIM,
            BLOCK = BLOCK,
            QUERY_HEAD_GROUPS = QUERY_HEAD_GROUPS,
            QUERY_EXPAND_DIM = QUERY_EXPAND_DIM,
            RETURN_SEL_GRADS = RETURN_SEL_GRADS,
            OFF_SEL_KV_BLOCKS = OFF_SEL_KV_BLOCKS,
            NUM_BLOCKS_PER_SEL = NUM_BLOCKS_PER_SEL,
            OFF_BLOCK_PER_SEL = OFF_BLOCK_PER_SEL,
            BLOCK_DV_USE_DOT = BLOCK_DV_USE_DOT,
            BLOCK_DK_USE_DOT = BLOCK_DK_USE_DOT,
        )

def native_sparse_attn_backward(
    do,
    q, page_table,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    kv_heads,
    seqlen_k,
    o,
    lse,
    dq, # dk, dv,
    do_slide = None,
    slide_out = None,
    slide_lse = None,
    block_size = 128,
    include_block_causal = True,
    return_sel_grads = False,
    sliding = False,
    block_dk_dv_use_dot = None
):
    # device = do.device

    # Make sure that the last dimension is contiguous
    if not is_contiguous(do):
        do = do.contiguous()

    if not is_contiguous(do_slide):
        do_slide = do_slide.contiguous()

    batch, q_heads, seqlen_q, dim = q.shape

    # _, kv_heads, seqlen_k, _ = k.shape
    assert divisible_by(q_heads, kv_heads)
    head_groups = q_heads // kv_heads
    assert divisible_by(16, head_groups)

    assert divisible_by(block_size, 16)

    num_blocks_per_sel = block_size // 16

    num_sel_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape

    # assert d in {16, 32, 64, 128}
    assert dim <= 128
    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)

    assert lse.shape == (batch, q_heads, seqlen_q_rounded)
    assert all([is_contiguous(t) for t in (q, page_table, o, dq)])

    softmax_scale = dim ** -0.5

    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)

    delta = torch.empty_like(lse)
    slide_delta = torch.empty_like(slide_lse)

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK"]), batch * q_heads)

    backward_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        q_heads,
        seqlen_q,
        seqlen_q_rounded,
        dim,
        BLOCK = block_size,
        BLOCK_HEADDIM = BLOCK_HEADDIM,
    )

    if sliding:
        backward_preprocess_do_o_dot[grid](
            slide_out,
            do_slide,
            slide_delta,
            slide_out.stride(0),
            slide_out.stride(1),
            slide_out.stride(2),
            do_slide.stride(0),
            do_slide.stride(1),
            do_slide.stride(2),
            q_heads,
            seqlen_q,
            seqlen_q_rounded,
            dim,
            BLOCK = block_size,
            BLOCK_HEADDIM = BLOCK_HEADDIM,
        )

    grid = lambda META: (
        int(include_block_causal) + int(sliding) + (num_sel_fine_blocks * num_blocks_per_sel),
        batch * kv_heads,
        triton.cdiv(seqlen_k, META['BLOCK'])
    )

    # ===============================================================
    # NOTE: 新增加
    stride_kb = kv_heads * 16 * dim
    stride_kh = 16 * dim
    stride_kn = dim
    stride_vb, stride_vh, stride_vn = stride_kb, stride_kh, stride_kn
    stride_dkb, stride_dkh, stride_dkn = stride_kb, stride_kh, stride_kn
    stride_dvb, stride_dvh, stride_dvn = stride_kb, stride_kh, stride_kn
    # ===============================================================

    backward_kernel[grid](
        q,
        page_table,
        # k,
        # v,
        kv_block_indices,
        kv_block_mask,
        kv_block_grads,
        do,
        dq,
        # dk,
        # dv,
        lse,
        delta,
        do_slide,
        slide_lse,
        slide_delta,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        # k.stride(0),
        # k.stride(1),
        # k.stride(2),
        # v.stride(0),
        # v.stride(1),
        # v.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        # dk.stride(0),
        # dk.stride(1),
        # dk.stride(2),
        # dv.stride(0),
        # dv.stride(1),
        # dv.stride(2),
        kv_block_indices.stride(0),
        kv_block_indices.stride(1),
        kv_block_indices.stride(2),
        lse.stride(0),
        delta.stride(0),
        kv_heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        dim,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        BLOCK_HEADDIM,
        BLOCK = 16,
        SEL_BLOCK = block_size,
        QUERY_HEAD_GROUPS = head_groups,
        EVEN_M = divisible_by(seqlen_q, block_size),
        EVEN_N = divisible_by(seqlen_k, block_size),
        EVEN_HEADDIM = BLOCK_HEADDIM == dim,
        RETURN_SEL_GRADS = return_sel_grads,
        INCLUDE_BLOCK_CAUSAL = include_block_causal,
        SLIDING = sliding,
        NUM_BLOCKS_PER_SEL = num_blocks_per_sel,
        BLOCK_DV_USE_DOT = default(block_dk_dv_use_dot, head_groups > 1),
        BLOCK_DK_USE_DOT = default(block_dk_dv_use_dot, head_groups > 1)
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )

    return delta, slide_delta

# native sparse attention function

from torch.autograd import Function

class NSA(Function):

    @classmethod
    def forward(
        self,
        ctx,
        q, k, v,
        sel_scale,
        manager,
        # block_size,
        # selected_block_indices,
        # fmask,
        # include_block_causal,
        # block_dk_dv_use_dot,
        # return_sliding_window_out
    ):
        # dtype = fq.dtype
        dtype = q.dtype

        q_heads = q.shape[1]
        assert divisible_by(q_heads, manager.num_kv_heads)
        head_groups = q_heads // manager.num_kv_heads

        q = q.half()

        out, slide_out, lse, slide_lse = native_sparse_attn_forward(
            q, 
            manager.page_table,
            manager.block_indices,
            manager.block_mask,
            manager.num_kv_heads,
            manager.num_kv,
            manager.block_size,
            # selected_block_indices,
            # fmask,
            # block_size = block_size,
            # include_block_causal = include_block_causal,
            # return_sliding_window_out = return_sliding_window_out
        )

        return_sel_grads = exists(sel_scale)

        if return_sel_grads:
            assert (sel_scale == 1.).all(), 'for now, must be straight through as multiplier of 1.'

        ctx.save_for_bcakward(q, out, slide_out, lse, slide_lse)
        ctx.manager = manager
        ctx.return_sel_grads = return_sel_grads

        # ctx.save_for_backward(fq, fk, fv, selected_block_indices, fmask, out, slide_out, lse, slide_lse)



        # ctx._saved_variables = (
        #     block_size,
        #     head_groups,
        #     return_sel_grads,
        #     include_block_causal,
        #     block_dk_dv_use_dot,
        #     return_sliding_window_out
        # )

        return out.type(dtype), slide_out.type(dtype), lse, slide_lse

        # return out.type(dtype), slide_out.type(dtype), lse, slide_lse

    @classmethod
    def backward(self, ctx, do, do_slide, _, __):
        device = do.device

        q, out, slide_out, lse, slide_lse = ctx.saved_tensors

        # (
        #     q, k, v,
        #     sel_block_indices,
        #     mask,
        #     out,
        #     slide_out,
        #     lse,
        #     slide_lse
        # ) = ctx.saved_tensors

        # (
        #     block_size,
        #     head_groups,
        #     return_sel_grads,
        #     include_block_causal,
        #     block_dk_dv_use_dot,
        #     return_sliding_window_out
        # ) = ctx._saved_variables

        do = do.half()
        do_slide = do_slide.half()

        dq = torch.zeros(q.shape, dtype = torch.float32, device = device)
        # dk = torch.zeros(k.shape, dtype = torch.float32, device = device)
        # dv = torch.zeros(v.shape, dtype = torch.float32, device = device)

        sel_grads = torch.zeros_like(ctx.manager.block_indices).float()

        native_sparse_attn_backward(
            do, q, 
            ctx.manager.page_table,
            ctx.manager.block_indices, 
            ctx.manager.block_mask, 
            sel_grads,
            ctx.manager.num_kv_heads,
            ctx.manager.num_kv,
            out, lse, dq, #dk, dv,
            do_slide = do_slide,
            slide_out = slide_out,
            slide_lse = slide_lse,
            block_size = ctx.manager.block_size,
            return_sel_grads = ctx.return_sel_grads,
            # include_block_causal = include_block_causal,
            # block_dk_dv_use_dot = block_dk_dv_use_dot,
            # sliding = return_sliding_window_out
        )
    
        ret_sel_grads = None

        if ctx.return_sel_grads:
            ret_sel_grads = sel_grads

        dk, dv = ctx.manager.grad

        return dq, dk, dv, ret_sel_grads, None

_native_sparse_attend = NSA.apply

# ein notation

# b - batch
# qh - query heads
# kh - key / value heads
# n - token sequence
# d - attention head dimension
# sel - selected indices

def native_sparse_attend(
    fq: Float['b qh n d'],
    fk: Float['b kh n d'],
    fv: Float['b kh n d'],
    block_size: int,
    selected_block_indices: Int['b qh n sel'] | Int['b kh n sel'],
    fmask: Bool['b qh n sel'] | Bool['b kh n sel'],
    sel_scale: Float['b kh n sel'] | Float['b qh n sel'] | None = None,
    include_block_causal = True,
    return_lse = False,
    block_dk_dv_use_dot = False,
    return_sliding_window_out = False
):
    seq_len = fq.shape[-2]
    q_heads, kv_heads, sel_heads = fq.shape[1], fk.shape[1], selected_block_indices.shape[1]

    assert divisible_by(q_heads, kv_heads)
    assert sel_heads in (q_heads, kv_heads)

    assert block_size >= 16, 'fine selection block size must be 16 or greater for now'

    # query heads within each group to attend to different segments

    if kv_heads != sel_heads:
        fk, fv = tuple(repeat(t, 'b h ... -> b (h gh) ...', gh = q_heads // kv_heads) for t in (fk, fv))

    out, sliding_out, lse, sliding_lse = _native_sparse_attend(
        fq, fk, fv,
        block_size,
        selected_block_indices,
        fmask,
        sel_scale,
        include_block_causal,
        block_dk_dv_use_dot,
        return_sliding_window_out
    )

    if return_sliding_window_out:
        out = (out, sliding_out)

    if not return_lse:
        return out

    lse = lse[..., :seq_len]
    sliding_lse = sliding_lse[..., :seq_len]

    if return_sliding_window_out:
        lse = (lse, sliding_lse)

    return out, lseh-attn'。将使用 PyTorch SDPA 作为参考。")
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            flash_attn_func = torch.nn.functional.scaled_dot_product_attention
        else:
            flash_attn_func = None

    # Helper functions for selection logic
    def pad_at_dim(t, pad, dim = -1, value = 0.):
        dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
        zeros = ((0, 0) * dims_from_right)
        return F.pad(t, (*zeros, *pad), value = value)
        
    def round_up_mult(n, mult):
        return ceil(n / mult) * mult

    page_size = 64
    batch_size = 1
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128

    num_kv_cache = 4096 
    num_new_toks = 1024
    
    total_kv_len = num_kv_cache + num_new_toks
    padded_kv_len = round_up_mult(total_kv_len, page_size)
    num_kv_blocks = padded_kv_len // page_size
    
    num_selected_blocks = 8
    num_query_blocks = num_new_toks // page_size

    k_cache = torch.randn((batch_size, num_kv_cache, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    v_cache = torch.randn((batch_size, num_kv_cache, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    
    q = torch.randn((batch_size, num_new_toks, num_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch_size, num_new_toks, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    v = torch.randn((batch_size, num_new_toks, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)

    k_cache.requires_grad_(True)
    v_cache.requires_grad_(True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    print("--- 开始测试：分页注意力 vs. 稀疏分页注意力 ---")

    # --- Generate selected_block_indices and selected_block_mask realistically ---
    with torch.no_grad():
        compress_block_size = page_size
        k_full_no_grad = torch.cat((k_cache, k), dim=1)
        
        padded_len = round_up_mult(total_kv_len, compress_block_size)
        k_padded = pad_at_dim(k_full_no_grad, (0, padded_len - total_kv_len), dim=1)
        k_padded_reshaped = rearrange(k_padded, 'b (w n) h d -> b w n h d', n=compress_block_size)
        ck = reduce(k_padded_reshaped, 'b w n h d -> b w h d', 'mean')

        q_for_coarse = rearrange(q, 'b n (h gh) d -> b n h gh d', gh=(num_heads // num_kv_heads))
        q_for_coarse = reduce(q_for_coarse, 'b n h gh d -> b n h d', 'mean')

        csim = einsum(q_for_coarse, ck, 'b i h d, b j h d -> b h i j')
        csim_q_blocked = rearrange(csim, 'b h (w n) j -> b h w n j', n=page_size)
        importance_scores = reduce(csim_q_blocked, 'b h w n j -> b h w j', 'mean')

        num_selectable_blocks = importance_scores.shape[-1]
        num_selected_to_take = min(num_selected_blocks, num_selectable_blocks)
        
        selected_values, selected_indices = importance_scores.topk(num_selected_to_take, dim=-1)
        
        if num_selected_to_take < num_selected_blocks:
            pad_width = num_selected_blocks - num_selected_to_take
            selected_indices = F.pad(selected_indices, (0, pad_width), value=0)

        selected_mask = torch.ones_like(selected_indices, dtype=torch.bool)
        if num_selected_to_take < num_selected_blocks:
            selected_mask[..., num_selected_to_take:] = False
            
    # --- 1. Sparse Paged Attention (Our Kernel) ---
    manager_sparse = CacheManager(batch_size, page_size, num_kv_heads, head_dim)
    k_full_padded = pad_at_dim(torch.cat((k_cache,k), dim=1), (0, padded_kv_len - total_kv_len), dim=1)
    v_full_padded = pad_at_dim(torch.cat((v_cache,v), dim=1), (0, padded_kv_len - total_kv_len), dim=1)
    manager_sparse.update(k_full_padded, v_full_padded)
    
    try:
        out_sparse = flash_paged_sparse_attn_func(q, k, v, selected_indices.contiguous(), selected_mask.contiguous(), manager_sparse, True)
        loss_sparse = out_sparse.sum()
        loss_sparse.backward()
        print("稀疏分页注意力（自定义核）运行成功。")
        dq_sparse = q.grad.clone()
        dk_sparse_full, dv_sparse_full = manager_sparse.grad
        
    except Exception as e:
        print(f"稀疏分页注意力（自定义核）运行失败: {e}")
        import traceback
        traceback.print_exc()
        out_sparse = None

    # --- 2. Masked SDPA (Reference) ---
    if flash_attn_func is not None and out_sparse is not None:
        q.grad, k.grad, v.grad, k_cache.grad, v_cache.grad = None, None, None, None, None
        
        # Create a full attention mask that mimics the sparse + causal pattern
        sparse_block_mask = torch.zeros(batch_size, num_kv_heads, num_query_blocks, num_kv_blocks, device=q.device, dtype=torch.bool)
        sparse_block_mask.scatter_(-1, selected_indices, True)
        attn_mask = repeat(sparse_block_mask, 'b h qb kb -> b h (qb ps_q) (kb ps_k)', ps_q=page_size, ps_k=page_size)
        attn_mask = attn_mask[:, :, :num_new_toks, :padded_kv_len]
        
        diag_causal_mask = torch.ones((num_new_toks, padded_kv_len), device=q.device, dtype=torch.bool).tril()
        
        q_block_indices = torch.arange(num_new_toks, device=q.device) // page_size
        k_block_indices = torch.arange(padded_kv_len, device=q.device) // page_size
        
        is_diag_block = q_block_indices[:, None] == k_block_indices[None, :]
        attn_mask = torch.where(is_diag_block[None, None, :, :], diag_causal_mask[None, None, :, :], attn_mask)

        # Expand for query heads
        attn_mask = repeat(attn_mask, 'b h ... -> b (h gh) ...', gh=(num_heads//num_kv_heads))
        
        try:
            # Use PyTorch's SDPA as the reference
            out_ref = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1,2), 
                k_full_padded.transpose(1,2).repeat_interleave(num_heads // num_kv_heads, dim=1),
                v_full_padded.transpose(1,2).repeat_interleave(num_heads // num_kv_heads, dim=1),
                attn_mask=attn_mask,
                is_causal=False).transpose(1,2)

            loss_ref = out_ref.sum()
            loss_ref.backward()

            print("带掩码的SDPA（参考）运行成功。")
            
            dq_ref = q.grad.clone()
            dk_ref = k_full_padded.grad.clone()
            dv_ref = v_full_padded.grad.clone()
            
            print("\n--- 结果比较 ---")
            print(f"Loss (Sparse Kernel): {loss_sparse.item()}")
            print(f"Loss (Reference SDPA): {loss_ref.item()}")
            print(f"Output dist: {torch.dist(out_sparse, out_ref)}")
            print(f"dq dist: {torch.dist(dq_sparse, dq_ref)}")
            print(f"dk dist: {torch.dist(dk_sparse_full, dk_ref)}")
            print(f"dv dist: {torch.dist(dv_sparse_full, dv_ref)}")
            
        except Exception as e:
            print(f"带掩码的SDPA（参考）运行失败: {e}")
            import traceback
            traceback.print_exc()

