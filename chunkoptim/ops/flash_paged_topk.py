import math

import torch
import triton
import triton.language as tl
from chunkoptim.ops.utils import IS_BF16_ATOM_ADD_SUPPORTED


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


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.jit
def _fwd_kernel_sparse(
    Q, T,
    selected_block_indices,
    Out, Lse,
    TMP,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_ob, stride_oh, stride_om,
    stride_kvh, stride_kvn_page, # Fixed: Removed stride_kvb
    stride_kvbl_b, stride_kvbl_h, stride_kvbl_q,
    nheads, seqlen_q, q_start_idx, headdim,
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
    sel_indices_ptrs = selected_block_indices + off_b * stride_kvbl_b + off_h * stride_kvbl_h + start_m_block * stride_kvbl_q
    q_idx = q_start_idx + offs_m

    # loop over the number of selected blocks for this query block
    for k_block_idx in tl.static_range(NUM_SEL_KV_BLOCKS):

        # load the logical index and mask of the selected key block
        kv_block_idx = tl.load(sel_indices_ptrs + k_block_idx)
        k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

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
        qk = tl.where(q_idx[:,None] >= k_idx[None,:], qk, float("-inf"))

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
    selected_block_indices, 
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    stride_kvh, stride_kvn_page, # Fixed: Removed stride_kvb
    stride_kvbl_b, stride_kvbl_h, stride_kvbl_q,
    nheads, seqlen_q, q_start_idx, headdim,
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

    sel_indices_ptrs = selected_block_indices + off_b * stride_kvbl_b + off_h * stride_kvbl_h + start_m_block * stride_kvbl_q

    q_idx = q_start_idx + offs_m

    for k_block_idx in range(NUM_SEL_KV_BLOCKS):
        kv_block_idx = tl.load(sel_indices_ptrs + k_block_idx)

        k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

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
        qk = tl.where(q_idx[:,None] >= k_idx[None,:], qk, float("-inf"))

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
        page_size: int,
        num_kv_heads: int,
        kv_head_dim: int,
        q_start_idx: int,
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
        selected_block_indices,
        o, lse,
        tmp,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        o.stride(0), o.stride(2), o.stride(1),
        stride_kvh, num_kv_heads * kv_head_dim,
        selected_block_indices.stride(0), selected_block_indices.stride(1), selected_block_indices.stride(2),
        nheads, seqlen_q, q_start_idx, d,
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
        page_size: int,
        num_kv_heads: int,
        kv_head_dim: int,
        q_start_idx: int,
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
        selected_block_indices,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        do.stride(0), do.stride(2), do.stride(1),
        dq.stride(0), dq.stride(2), dq.stride(1),
        stride_kvh, num_kv_heads * kv_head_dim, # stride_kvn_page
        selected_block_indices.stride(0), selected_block_indices.stride(1), selected_block_indices.stride(2),
        nheads, seqlen_q, q_start_idx, d,
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
            manager):

        q = q if q.stride(-1) == 1 else q.contiguous()

        manager.update_top_indices(q)
        manager.cuda_in_forward()

        o, lse, ctx.softmax_scale = _flash_attn_forward_sparse(
            q, page_table=manager.page_table,
            selected_block_indices=manager.top_indices, 
            page_size=manager.page_size, 
            num_kv_heads=manager.num_kv_heads, 
            kv_head_dim=manager.head_dim,
            q_start_idx=manager.num_kv - q.shape[1],
            softmax_scale=None)

        manager.cpu_in_forward()
        ctx.save_for_backward(q, o, lse)
        ctx.manager = manager

        return o

    @staticmethod
    def backward(ctx, do):
        q, o, lse = ctx.saved_tensors

        dq = torch.zeros_like(q)
        
        ctx.manager.cuda_in_backward()

        _flash_attn_backward_sparse(
            o, do, q, dq,
            ctx.manager.page_table,
            ctx.manager.top_indices, 
            ctx.manager.page_size,
            ctx.manager.num_kv_heads,
            ctx.manager.head_dim,
            ctx.manager.num_kv - q.shape[1],
            lse,
            ctx.softmax_scale)
        
        dk, dv = ctx.manager.grad

        ctx.manager.cpu_in_backward()

        return dq, dk, dv, None, None, None


flash_paged_sparse_attn_func = FlashPagedSparseAttn.apply