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

IS_BF16_ATOM_ADD_SUPPORTED = triton.__version__ >= "3.4.0"

if not IS_BF16_ATOM_ADD_SUPPORTED:
    print(colorize('yellow', "[flash_paged_attn.py]: BF16 atomic add is not supported by Triton < 3.4.0, please upgrade Triton to 3.4.0 or later."), flush=True)
    print(colorize('yellow', ">>>") + ' ' + "Press Enter to continue, or Ctrl+C to exit...", flush=True)

# =================================================================================
# Original Paged Attention Kernels
# =================================================================================

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
    stride_kvh, stride_kvn_page, # Fixed: Removed stride_kvb
    stride_kvbl_b, stride_kvbl_h, stride_kvbl_q,
    nheads, seqlen_q, headdim,
    seqlen_q_rounded, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE

    # offsets for the current query block
    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # pointers to Q
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])

    # online softmax stats
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # load Q for the entire block
    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    # pointers to the selected block indices and mask for this query block
    sel_indices_ptrs = selected_block_indices + off_b * stride_kvbl_b + off_kv_h * stride_kvbl_h + start_m_block * stride_kvbl_q
    sel_mask_ptrs = selected_block_mask + off_b * stride_kvbl_b + off_kv_h * stride_kvbl_h + start_m_block * stride_kvbl_q

    # loop over the number of selected blocks for this query block
    for k_block_idx in range(NUM_SEL_KV_BLOCKS):
        # load the logical index and mask of the selected key block
        kv_block_idx = tl.load(sel_indices_ptrs + k_block_idx)
        block_mask = tl.load(sel_mask_ptrs + k_block_idx)

        # get page pointers from the page table (T)
        k_page_ptr = tl.load(T + kv_block_idx * 4)
        v_page_ptr = tl.load(T + kv_block_idx * 4 + 1)
        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))

        # construct pointers to K and V pages
        offs_n = tl.arange(0, BLOCK_N)
        # Fixed: a page pointer is physical and batch-agnostic. The offset should only be for the head.
        kv_page_offs = off_kv_h * stride_kvh
        k_ptrs = k_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        v_ptrs = v_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])

        # load K and V blocks
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # compute QK scores
        qk = tl.dot(q, k.T)
        
        # apply the runtime mask
        qk = tl.where(block_mask, qk, float("-inf"))

        # update softmax stats
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, m_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # update accumulator
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # update log-sum-exp
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # write back output and lse
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
def _bwd_kernel_sparse(
    Q, DO, DQ,
    T, LSE, D,
    selected_block_indices, selected_block_mask,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    stride_kvh, stride_kvn_page, # Fixed: Removed stride_kvb
    stride_kvbl_b, stride_kvbl_h, stride_kvbl_q,
    nheads, seqlen_q, headdim,
    seqlen_q_rounded, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
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

    sel_indices_ptrs = selected_block_indices + off_b * stride_kvbl_b + off_kv_h * stride_kvbl_h + start_m_block * stride_kvbl_q
    sel_mask_ptrs = selected_block_mask + off_b * stride_kvbl_b + off_kv_h * stride_kvbl_h + start_m_block * stride_kvbl_q

    for k_block_idx in range(NUM_SEL_KV_BLOCKS):
        kv_block_idx = tl.load(sel_indices_ptrs + k_block_idx)
        block_mask = tl.load(sel_mask_ptrs + k_block_idx)

        k_page_ptr = tl.load(T + kv_block_idx * 4)
        v_page_ptr = tl.load(T + kv_block_idx * 4 + 1)
        dk_page_ptr = tl.load(T + kv_block_idx * 4 + 2)
        dv_page_ptr = tl.load(T + kv_block_idx * 4 + 3)

        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))
        dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))
        dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))

        offs_n = tl.arange(0, BLOCK_N)
        # Fixed: a page pointer is physical and batch-agnostic. The offset should only be for the head.
        kv_page_offs = off_kv_h * stride_kvh
        k_ptrs = k_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        v_ptrs = v_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        dk_ptrs = dk_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        dv_ptrs = dv_page_ptr + kv_page_offs + (offs_n[:, None] * stride_kvn_page + offs_d[None, :])
        
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.dot(q, k.T)
        qk = tl.where(block_mask, qk, float("-inf"))
        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        dv_block = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_ptrs, dv_block, sem='relaxed')

        dp = tl.dot(do, v.T)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        dk_block = tl.dot(ds.T, q)
        tl.atomic_add(dk_ptrs, dk_block, sem='relaxed')

        dq_block += tl.dot(ds, k)

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
        causal=False, # Causal masking within blocks is not implemented for simplicity
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

    # Fixed: Correctly define stride_kvh for in-page head navigation
    stride_kvh = kv_head_dim

    _fwd_kernel_sparse[grid](
        q, page_table,
        selected_block_indices, selected_block_mask,
        o, lse,
        tmp,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        o.stride(0), o.stride(2), o.stride(1),
        stride_kvh, num_kv_heads * kv_head_dim, # stride_kvn_page
        selected_block_indices.stride(0), selected_block_indices.stride(1), selected_block_indices.stride(2),
        nheads, seqlen_q, d,
        seqlen_q_rounded, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=page_size,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=(seqlen_q % BLOCK_M == 0),
        NUM_SEL_KV_BLOCKS=num_sel,
        GROUP_SIZE=GROUP_SIZE,
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
        softmax_scale: float
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
    
    # Fixed: Correctly define stride_kvh for in-page head navigation
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
        stride_kvh, num_kv_heads * kv_head_dim, # stride_kvn_page
        selected_block_indices.stride(0), selected_block_indices.stride(1), selected_block_indices.stride(2),
        nheads, seqlen_q, d,
        seqlen_q_rounded, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=page_size,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=(seqlen_q % BLOCK_M == 0),
        NUM_SEL_KV_BLOCKS=num_sel,
        GROUP_SIZE=GROUP_SIZE,
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
            manager):

        q = q if q.stride(-1) == 1 else q.contiguous()

        o, lse, ctx.softmax_scale = _flash_attn_forward_sparse(
            q, manager.page_table,
            selected_block_indices, selected_block_mask,
            manager.page_size, manager.num_kv_heads, manager.head_dim,
            causal=True, softmax_scale=None)

        ctx.save_for_backward(q, o, lse, selected_block_indices, selected_block_mask)
        ctx.manager = manager
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
                ctx.softmax_scale)

            dk, dv = ctx.manager.grad

        return dq, dk, dv, None, None, None


flash_paged_sparse_attn_func = FlashPagedSparseAttn.apply


if __name__ == '__main__':

    from profiler import WallTime

    profile_page = WallTime("page", cuda=0)
    profile_nsa = WallTime("nsa", cuda=0)

    # A simple check to ensure 'chunkoptim' is available or provide a mock
    try:
        from chunkoptim.kv_cache import CacheManager
    except ImportError:
        print("Warning: 'chunkoptim' not found. Using a mock CacheManager for testing.")
        class CacheManager:
            def __init__(self, batch_size, page_size, num_kv_heads, head_dim):
                self.batch_size = batch_size
                self.page_size = page_size
                self.num_kv_heads = num_kv_heads
                self.head_dim = head_dim
                self.reset()

            def update(self, k, v):
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
                # This is a simplification. A real implementation would need to handle grads correctly.
                dk = torch.cat([p.flatten(0,1) for p in self.all_dk_pages], dim=0) if self.all_dk_pages else None
                dv = torch.cat([p.flatten(0,1) for p in self.all_dv_pages], dim=0) if self.all_dv_pages else None
                return dk, dv

    # A simple check to ensure 'flash_attn' is available or provide a mock
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print("Warning: 'flash-attn' not found. Skipping reference comparison.")
        flash_attn_func = None

    page_size = 64
    batch_size = 1
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128

    num_kv_cache = 4096 
    num_new_toks = 1024

    total_kv_len = num_kv_cache + num_new_toks
    num_kv_blocks = (total_kv_len + page_size - 1) // page_size

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

    print("--- Testing Paged Attention vs. Sparse Paged Attention ---")

    # --- Standard Paged Attention ---
    manager_dense = CacheManager(batch_size, page_size, num_kv_heads, head_dim)
    manager_dense.update(k_cache, v_cache)
    manager_dense.update(k, v)
    
    try:
        for _ in range(10):
            with profile_page:
                out_dense = flash_paged_attn_func(q, k, v, manager_dense)
        loss_dense = out_dense.sum()
        loss_dense.backward()
        print("Standard Paged Attention ran successfully.")
        dq_dense, (dk_dense, dv_dense) = q.grad.clone(), manager_dense.grad
        
    except Exception as e:
        print(f"Standard Paged Attention failed: {e}")
        out_dense = None

    # --- Sparse Paged Attention ---
    q.grad, k.grad, v.grad, k_cache.grad, v_cache.grad = None, None, None, None, None
    manager_sparse = CacheManager(batch_size, page_size, num_kv_heads, head_dim)
    manager_sparse.update(k_cache, v_cache)
    manager_sparse.update(k, v)

    selected_indices = torch.arange(num_kv_blocks - num_selected_blocks, num_kv_blocks, device='cuda').long()
    selected_indices = selected_indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, num_kv_heads, num_query_blocks, num_selected_blocks).contiguous()
    selected_mask = torch.ones_like(selected_indices, dtype=torch.bool).contiguous()
    
    try:
        for _ in range(10):
            with profile_nsa:
                out_sparse = flash_paged_sparse_attn_func(q, k, v, selected_indices, selected_mask, manager_sparse)
        loss_sparse = out_sparse.sum()
        loss_sparse.backward()
        print("Sparse Paged Attention ran successfully.")
        dq_sparse, (dk_sparse, dv_sparse) = q.grad.clone(), manager_sparse.grad
    except Exception as e:
        print(f"Sparse Paged Attention failed: {e}")
        out_sparse = None

    # --- Comparison ---
    if out_dense is not None and out_sparse is not None:
        print("\n--- Comparison Results ---")
        print(f"Output difference (should be non-zero): {torch.dist(out_dense, out_sparse)}")
        
        if flash_attn_func:
            k_full = torch.cat((k_cache, k), dim=1)
            v_full = torch.cat((v_cache, v), dim=1)
            
            k_selected_for_ref = []
            v_selected_for_ref = []
            for i in range(num_selected_blocks):
                block_idx = num_kv_blocks - num_selected_blocks + i
                start, end = block_idx * page_size, (block_idx + 1) * page_size
                k_selected_for_ref.append(k_full[:, start:end])
                v_selected_for_ref.append(v_full[:, start:end])

            k_ref_sparse = torch.cat(k_selected_for_ref, dim=1)
            v_ref_sparse = torch.cat(v_selected_for_ref, dim=1)

            out_ref_sparse = flash_attn_func(q, k_ref_sparse, v_ref_sparse, causal=False)

            print(f"Sparse output vs. manual sparse reference difference: {torch.dist(out_sparse, out_ref_sparse)}")

    profile_nsa.result(detail=True)
    profile_page.result(detail=True)