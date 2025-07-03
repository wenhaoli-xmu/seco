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
    I,
    num_pages,
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
    stride_ib,
    stride_ih,
    stride_im,
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

    # (BLOCK_M, 1, BLOCK_DIM)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None, None] * stride_qm + offs_d[None, None, :])
    
    # (BLOCK_N, BLOCK_DIM)
    kv_offs = off_b * stride_kvb + (off_h // GROUP_SIZE) * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])

    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M & EVEN_N:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None, None] < seqlen_q, other=0.0)
        
    i_ptrs = I + off_b * stride_ib + off_h * stride_ih + offs_m * stride_im
    seqlen_q_mask = offs_m < seqlen_q
        
    for n_idx in range(num_pages):

        # (BLOCK_M,)
        page_indices = tl.load(
            i_ptrs + n_idx, 
            mask=seqlen_q_mask)
        
        # (BLOCK_M, BLOCK_N)
        seqlen_k_mask = (page_indices[:, None] * BLOCK_N + offs_n[None, :]) < seqlen_k

        # (BLOCK_M, BLOCK_N)
        causal_mask = (seqlen_k - seqlen_q + offs_m[:, None]) >= (page_indices[:, None] * BLOCK_N + offs_n[None, :])

        # (BLOCK_M,)
        k_page_ptrs = tl.load(T + page_indices * 4)
        v_page_ptrs = tl.load(T + page_indices * 4 + 1)

        k_page_ptrs = tl.cast(k_page_ptrs, tl.pointer_type(tl.bfloat16))
        v_page_ptrs = tl.cast(v_page_ptrs, tl.pointer_type(tl.bfloat16))

        # (BLOCK_M, BLOCK_N, BLOCK_DIM)
        k_ptrs = k_page_ptrs[:, None, None] + kv_offs[None, :, :]
        v_ptrs = v_page_ptrs[:, None, None] + kv_offs[None, :, :]

        if EVEN_N & EVEN_M:
            k = tl.load(k_ptrs)
        else:
            # (BLOCK_M, BLOCK_N, BLOCK_DIM)
            k = tl.load(
                k_ptrs,
                mask=seqlen_k_mask[:, :, None] & seqlen_q_mask[:, None, None],
                other=0.0,)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # q: (m,d)
        # k: (m,n,d)
        qk += tl.sum(q * k, axis=2)

        if not EVEN_N:
            # (BLOCK_M, BLOCK_N)
            qk += tl.where(seqlen_k_mask, 0, float("-inf"))

        if IS_CAUSAL:
            qk = tl.where(causal_mask, qk, float("-inf"))

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
            # (BLOCK_M, BLOCK_N, BLOCK_DIM)
            v = tl.load(
                v_ptrs,
                mask=seqlen_k_mask[:, :, None] & seqlen_q_mask[:, None, None],
                other=0.0)

        # (BLOCK_M, 1, BLOCK_DIM)
        p = p.to(v.dtype).expand_dims(axis=2)

        # (BLOCK_M, BLOCK_DIM)
        acc_o += tl.sum(p * v, axis=1)

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
    seqlen_k_mask,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    if EVEN_N & EVEN_M:
        tl.atomic_add(dv_ptrs, dv)
        tl.atomic_add(dk_ptrs, dk)
    else:
        tl.atomic_add(dv_ptrs, dv, mask=seqlen_k_mask[:, :, None])
        tl.atomic_add(dk_ptrs, dk, mask=seqlen_k_mask[:, :, None])


@triton.jit
def _bwd_kernel_one_col_block(
    n_idx,
    T,
    I,
    stride_im,
    kv_off_base,
    Q,
    DO,
    DQ,
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
    IS_BF16_ATOM_ADD_SUPPORTED: tl.constexpr,
):
    begin_m = 0

    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    # offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # (BLOCK_N, BLOCK_DIM)
    kv_offs = kv_off_base + tl.arange(0, BLOCK_N)[:, None] * stride_kvn + offs_d[None, :]

    q_ptrs = Q + (offs_qm[:, None, None] * stride_qm + offs_d[None, None, :])

    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])

    if begin_m >= seqlen_q:
        tl.device_assert(False)
        tl.debug_barrier()

    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        # (m,)
        page_indices = tl.load(
            I + offs_m * stride_im + n_idx,
            mask=offs_m_curr < seqlen_q)
        
        # (m,n)
        seqlen_k_mask = (page_indices[:, None] * BLOCK_N + offs_n[None, :]) < seqlen_k

        # (m,n)
        causal_mask = (seqlen_k - seqlen_q + offs_m_curr[:, None]) >= (page_indices[:, None] * BLOCK_N + offs_n[None, :])

        # (m,n)
        k_page_ptrs = tl.load(T + page_indices * 4)
        v_page_ptrs = tl.load(T + page_indices * 4 + 1)
        dk_page_ptrs = tl.load(T + page_indices * 4 + 2)
        dv_page_ptrs = tl.load(T + page_indices * 4 + 3)

        k_page_ptrs = tl.cast(k_page_ptrs, tl.pointer_type(tl.bfloat16))
        v_page_ptrs = tl.cast(v_page_ptrs, tl.pointer_type(tl.bfloat16))
        dk_page_ptrs = tl.cast(dk_page_ptrs, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))
        dv_page_ptrs = tl.cast(dv_page_ptrs, tl.pointer_type(tl.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else tl.float32))

        # (m,n,d)
        k_ptrs = k_page_ptrs[:, None, None] + kv_offs[None, :, :]
        v_ptrs = v_page_ptrs[:, None, None] + kv_offs[None, :, :]
        dk_ptrs = dk_page_ptrs[:, None, None] + kv_offs[None, :, :]
        dv_ptrs = dv_page_ptrs[:, None, None] + kv_offs[None, :, :]

        if EVEN_N & EVEN_M:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            # (m,n,d)
            k = tl.load(k_ptrs, mask=seqlen_k_mask[:, :, None], other=0.0)
            v = tl.load(v_ptrs, mask=seqlen_k_mask[:, :, None], other=0.0)

        if EVEN_M:
            q = tl.load(q_ptrs)
        else:
            # (m,1,d)
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None, None] < seqlen_q, other=0.0)

        # (m,n)
        qk = tl.sum(q * k, axis=2)

        if not EVEN_N:
            qk = tl.where(seqlen_k_mask, qk, float("-inf"))

        if IS_CAUSAL:
            qk = tl.where(causal_mask, qk, float("-inf"))

        if not EVEN_M:
            tl.debug_barrier()

        lse_i = tl.load(LSE + offs_m_curr)

        # (m,n)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        if EVEN_M:
            do = tl.load(do_ptrs)
        else:
            # (m,1,d)
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                other=0.0)

        # p: (m,n)
        # do: (m,d)
        # dv: (m,n,d)
        dv = p.to(do.dtype).expand_dims(2) * do.expand_dims(1)

        if not EVEN_M:
            tl.debug_barrier()

        # (BLOCK_M, BLOCK_N)
        # do: (BLOCK_M, 1, BLOCK_DIM)
        # v: (BLOCK_M, BLOCK_N, BLOCK_DIM)
        dp = tl.sum(do * v, axis=2)

        Di = tl.load(D + offs_m_curr)

        # (BLOCK_M, BLOCK_N)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        # ds: (m,n)
        # q: (m,d)
        # dk: (m,n,d)
        dk = ds.expand_dims(2) * q

        if not EVEN_M: 
            tl.debug_barrier()

        if not ATOMIC_ADD:
            if EVEN_M:  # Race condition if we just do EVEN_M
                # (m,d)
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")

                # ds: (m,n,1)
                # k: (m,n,d)
                # dq: (m,d)
                dq += tl.sum(ds.expand_dims(2) * k, axis=1)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                dq = tl.load(
                    dq_ptrs,
                    mask=offs_m_curr[:, None] < seqlen_q,
                    other=0.0,
                    eviction_policy="evict_last")
                
                dq += tl.sum(ds.expand_dims(2) * k, axis=1)

                tl.store(
                    dq_ptrs,
                    dq,
                    mask=offs_m_curr[:, None] < seqlen_q,
                    eviction_policy="evict_last")

        else:
            dq = tl.sum(ds.expand_dims(2) * k, axis=1)
            if EVEN_M:
                tl.atomic_add(dq_ptrs, dq)
            else:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)

        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

        _bwd_store_dk_dv(
            dk_ptrs, 
            dv_ptrs,
            dk, # (m,n,d)
            dv, # (m,n,d)
            seqlen_k_mask,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.jit
def _bwd_kernel(
    Q,
    DO,
    DQ,
    T,
    I,
    num_pages,
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
    stride_ib,
    stride_ih,
    stride_im,
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

    kv_offs = off_b * stride_kvb + (off_h // GROUP_SIZE) * stride_kvh

    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded

    I += off_b * stride_ib + off_h * stride_ih

    if not SEQUENCE_PARALLEL:

        for n_idx in range(num_pages):

            _bwd_kernel_one_col_block(
                n_idx,
                T,
                I,
                stride_im,
                kv_offs,
                Q,
                # k_ptrs,
                # v_ptrs,
                DO,
                DQ,
                # dk_ptrs,
                # dv_ptrs,
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
                BLOCK_N=BLOCK_N,
                IS_BF16_ATOM_ADD_SUPPORTED=IS_BF16_ATOM_ADD_SUPPORTED)
    else:
        n_idx = tl.program_id(0)

        _bwd_kernel_one_col_block(
            n_idx,
            T,
            I,
            stride_im,
            kv_offs,
            Q,
            # k_ptrs,
            # v_ptrs,
            DO,
            DQ,
            # dk_ptrs,
            # dv_ptrs,
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
            BLOCK_N=BLOCK_N,
            IS_BF16_ATOM_ADD_SUPPORTED=IS_BF16_ATOM_ADD_SUPPORTED)


def _flash_attn_forward(
        q: torch.Tensor, 
        page_table: torch.Tensor,
        page_indices: torch.Tensor,
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

    BLOCK_HEADDIM = d
    BLOCK = page_size
    GROUP_SIZE = nheads // num_kv_heads

    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    seqlen_k = num_kv
    num_select_pages = page_indices.shape[-1]

    _fwd_kernel[grid](
        q,
        page_table,
        page_indices,
        num_select_pages,
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
        page_indices.stride(0),
        page_indices.stride(1),
        page_indices.stride(2),
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
        page_indices: torch.Tensor,
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
        num_select_pages if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,)
    
    seqlen_k = num_kv

    num_select_pages = page_indices.shape[-1]

    _bwd_kernel[grid](
        q,
        do,
        dq if IS_BF16_ATOM_ADD_SUPPORTED else dq_accum,
        page_table,
        page_indices,
        num_select_pages,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        # page table strides
        page_size * num_kv_heads * kv_head_dim,
        kv_head_dim,
        num_kv_heads * kv_head_dim,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq.stride(0),
        dq.stride(2),
        dq.stride(1),
        page_indices.stride(0),
        page_indices.stride(1),
        page_indices.stride(2),
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


class FlashPagedMoBA(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            manager):

        q = q if q.stride(-1) == 1 else q.contiguous()
        
        page_indices = manager.select_blocks(q)

        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, 
            manager.page_table,
            page_indices,
            manager.num_kv, 
            manager.page_size, 
            manager.num_kv_heads, 
            manager.head_dim,
            bias=None, 
            causal=True, 
            softmax_scale=None)
        
        ctx.save_for_backward(q, o, lse)
        ctx.page_indices = page_indices
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
                ctx.page_indices,
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


flash_paged_moba = FlashPagedMoBA.apply


if __name__ == '__main__':
    from flash_attn import flash_attn_func
    from chunkoptim.cache.sparse_cache import MoBACacheManager
    page_size = 16

    num_heads = 32
    num_kv_heads = 4

    num_kv_cache = 12800
    num_new_toks = 1024

    k_cache1 = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    v_cache1 = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    k_cache2 = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    v_cache2 = torch.randn((1, num_kv_cache, num_kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    manager = MoBACacheManager(1, page_size, num_kv_heads, 128, 1_000_000_000)

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
    ref_time_fwd = WallTime('ref-fwd', cuda=0)
    our_time_fwd = WallTime('our-fwd', cuda=0)
    ref_time_bwd = WallTime('ref-bwd', cuda=0)
    our_time_bwd = WallTime('our-bwd', cuda=0)

    for _ in range(20):

        q.grad, k.grad, v.grad, k_cache1.grad, v_cache1.grad, k_cache2.grad, v_cache2.grad = None, None, None, None, None, None, None
        
        with ref_time_fwd:
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

        with our_time_fwd:
            our_bwd = flash_paged_moba(q, k, v, manager)
        
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
    ref_time_fwd.result(detail=True)
    our_time_fwd.result(detail=True)
