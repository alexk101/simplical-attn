# Source: https://arxiv.org/pdf/2507.02754
# 2-SIMPLICIAL ATTENTION - TRITON KERNELS

import triton
from triton import Config
import triton.language as tl
import torch


# Forward pass
@triton.autotune(configs=[Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 32, "num_stages": 1}, num_warps=4)], key=["HEAD_DIM"])
@triton.jit
def two_simplicial_attn_fwd_kernel(
    Q_ptr, # [b, s, k, h]
    K1_ptr, # [b, s, k, h]
    K2_ptr, # [b, s, k, h]
    V1_ptr, # [b, s, k, h]
    V2_ptr, # [b, s, k, h]
    O_ptr, # [b, s, k, h]
    M_ptr, # [b, k, s]
    D_ptr, # [b, k, s] - row-wise gradient info for backward pass
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1: tl.constexpr,
    w2: tl.constexpr,
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    out_stride_b,
    out_stride_s,
    out_stride_k,
    out_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    num_stages: tl.constexpr,
):
    data_dtype = tl.bfloat16
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    q_start = tl.program_id(0) * BLOCK_SIZE_Q
    q_end = q_start + BLOCK_SIZE_Q
    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k

    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk
    O_ptr += qkv_offs_bk
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k

    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype) - float("inf")
    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype)
    acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=compute_dtype)

    q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    q_mask_s = q_offs_s < seq_len
    qkv_mask_h = qkv_offs_h < head_dim
    q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h
    q_mask = q_mask_s[:, None] & (qkv_mask_h[None, :])

    q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(compute_dtype) # [BLOCK_SIZE_Q, HEAD_DIM]
    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)

    for kv1_idx in tl.range(tl.maximum(0, q_start - w1), tl.minimum(seq_len, q_end)):
        k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
        k1_tile = (tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(compute_dtype))[None, :] # [1, HEAD_DIM]
        qk1 = q_tile * k1_tile # [BLOCK_SIZE_Q, HEAD_DIM]
        qk1 = qk1.to(gemm_dtype)

        v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
        v1_tile = (tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(compute_dtype))[None, :] # [1, HEAD_DIM]

        for kv2_idx in tl.range(tl.maximum(0, q_start - w2), tl.minimum(seq_len, q_end), BLOCK_SIZE_KV, num_stages=num_stages):
            kv2_offs_s = kv2_idx + tl.arange(0, BLOCK_SIZE_KV)
            kv2_mask_s = kv2_offs_s < seq_len
            k2t_mask = kv2_mask_s[None, :] & qkv_mask_h[:, None]
            v2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]
            k2_offs = (kv2_offs_s[None, :] * k2_stride_s + qkv_offs_h[:, None] * k2_stride_h)
            v2_offs = (kv2_offs_s[:, None] * v2_stride_s + qkv_offs_h[None, :] * v2_stride_h)
            k2t_tile = tl.load(K2_ptr + k2_offs, mask=k2t_mask).to(compute_dtype) # [HEAD_DIM, BLOCK_SIZE_KV]
            v2_tile = tl.load(V2_ptr + v2_offs, mask=v2_mask).to(compute_dtype) # [BLOCK_SIZE_KV, HEAD_DIM]
            k2t_tile += K2_BIAS
            v2_tile += V2_BIAS
            k2t_tile = k2t_tile.to(gemm_dtype)
            v2_tile = v2_tile.to(compute_dtype)

            qk = tl.dot(
                qk1 * softmax_scale,
                k2t_tile,
                input_precision="tf32", # INPUT_PRECISION,
                out_dtype=tl.float32,
            ) # [BLOCK_SIZE_Q, BLOCK_SIZE_KV]

            qk_mask = q_mask_s[:, None] & kv2_mask_s[None, :]
            # Mask for q_idx - w1 < kv1_idx <= q_idx
            # and q_idx - w2 < kv2_offs_s <= q_idx
            kv1_local_mask = ((q_offs_s[:, None] - w1) < kv1_idx) & (kv1_idx <= q_offs_s[:, None])
            kv2_local_mask = ((q_offs_s[:, None] - w2) < kv2_offs_s[None, :]) & (kv2_offs_s[None, :] <= q_offs_s[:, None])
            qk_mask &= kv1_local_mask & kv2_local_mask
            qk += tl.where(qk_mask, 0, -1.0e38)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v12_tile = v1_tile * v2_tile # [BLOCK_SIZE_KV, HEAD_DIM]
            acc += tl.dot(
                p.to(gemm_dtype),
                v12_tile.to(gemm_dtype),
                input_precision="ieee", # INPUT_PRECISION,
                out_dtype=tl.float32,
            )

            m_i = m_ij

    # Store the final result after all accumulation is done
    acc = acc / l_i[:, None]
    acc = tl.where(q_mask, acc, 0.0)
    acc_store = acc.to(data_dtype)  # Use temporary variable to avoid type reassignment
    out_offs = q_offs_s[:, None] * out_stride_s + qkv_offs_h[None, :] * out_stride_h
    tl.store(O_ptr + out_offs, acc_store, mask=q_mask)

    m = m_i + tl.log(l_i)
    m_offs = q_offs_s * m_stride_s
    m_mask = q_offs_s < seq_len
    tl.store(M_ptr + m_offs, m, mask=m_mask)
    
    # D will be computed in backward pass preprocessing, just store placeholder zeros for now
    d_offs = q_offs_s * d_stride_s  
    tl.store(D_ptr + d_offs, tl.zeros_like(m_i), mask=m_mask)


# Backward pass
@triton.autotune(configs=[Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 32}, num_warps=4)], key=["HEAD_DIM"])
@triton.jit
def two_simplicial_attn_bwd_kv1_kernel(
    Q_ptr, # [b, s, k, h]
    K1_ptr, # [b, s, k, h]
    K2_ptr, # [b, s, k, h]
    V1_ptr, # [b, s, k, h]
    V2_ptr, # [b, s, k, h]
    dO_ptr, # [b, s, k, h]
    M_ptr, # [b, k, s]
    D_ptr, # [b, k, s]
    dQ_ptr, # [b, s, k, h]
    dK1_ptr, # [b, s, k, h]
    dV1_ptr, # [b, s, k, h]
    # Skip writing dk2, dv2 for now.
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1, # Q[i]: KV1(i-w1,i]
    w2, # Q[i]: KV2(i-w2,i]
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    dq_stride_b,
    dq_stride_s,
    dq_stride_k,
    dq_stride_h,
    dk1_stride_b,
    dk1_stride_s,
    dk1_stride_k,
    dk1_stride_h,
    dv1_stride_b,
    dv1_stride_s,
    dv1_stride_k,
    dv1_stride_h,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    COMPUTE_DQ: tl.constexpr,
    is_flipped: tl.constexpr,
):
    data_dtype = tl.bfloat16
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    kv1_start = tl.program_id(0) * BLOCK_SIZE_KV
    kv1_end = kv1_start + BLOCK_SIZE_KV
    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk
    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
    dK1_ptr += offs_b * dk1_stride_b + offs_k * dk1_stride_k
    dV1_ptr += offs_b * dv1_stride_b + offs_k * dv1_stride_k
    if COMPUTE_DQ:
        dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k

    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    qkv_mask_h = qkv_offs_h < head_dim

    kv1_offs_s = kv1_start + tl.arange(0, BLOCK_SIZE_KV)

    k1_offs = kv1_offs_s[:, None] * k1_stride_s + qkv_offs_h[None, :] * k1_stride_h
    kv1_mask_s = kv1_offs_s < seq_len
    kv1_mask = kv1_mask_s[:, None] & qkv_mask_h[None, :]
    k1_tile = tl.load(K1_ptr + k1_offs, mask=kv1_mask).to(compute_dtype) # [BLOCK_SIZE_KV, HEAD_DIM]
    v1_offs = kv1_offs_s[:, None] * v1_stride_s + qkv_offs_h[None, :] * v1_stride_h
    v1_tile = tl.load(V1_ptr + v1_offs, mask=kv1_mask).to(compute_dtype) # [BLOCK_SIZE_KV, HEAD_DIM]
    if is_flipped:
        k1_tile += K2_BIAS
        v1_tile += V2_BIAS
    dv1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)
    dk1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)

    # for kv2_idx in tl.range(0, seq_len):
    # kv1 - w2 < kv2 <= kv1 + w1
    for kv2_idx in tl.range(tl.maximum(0, kv1_start - w2), tl.minimum(seq_len, kv1_end + w1)):
        k2_offs = kv2_idx * k2_stride_s + qkv_offs_h * k2_stride_h
        k2_tile = (tl.load(K2_ptr + k2_offs, mask=qkv_mask_h).to(compute_dtype))[None, :] # [1, HEAD_DIM]
        v2_offs = kv2_idx * v2_stride_s + qkv_offs_h * v2_stride_h
        v2_tile = (tl.load(V2_ptr + v2_offs, mask=qkv_mask_h).to(compute_dtype))[None, :] # [1, HEAD_DIM]
        if not is_flipped:
            k2_tile += K2_BIAS
            v2_tile += V2_BIAS
        k1k2 = k1_tile * k2_tile # [BLOCK_SIZE_KV, HEAD_DIM]
        v1v2 = v1_tile * v2_tile # [BLOCK_SIZE_KV, HEAD_DIM]
        k1k2 = k1k2.to(gemm_dtype)
        v1v2 = v1v2.to(gemm_dtype)

        # kv1 <= q < kv1 + w1
        # kv2 <= q < kv2 + w2
        q_start = tl.maximum(kv1_start, kv2_idx)
        q_end = tl.minimum(seq_len, tl.minimum(kv1_end + w1, kv2_idx + w2))
        for q_idx in tl.range(q_start, q_end, BLOCK_SIZE_Q):
            # Load qt, m, d, dO
            q_offs_s = q_idx + tl.arange(0, BLOCK_SIZE_Q)
            q_offs = q_offs_s[None, :] * q_stride_s + qkv_offs_h[:, None] * q_stride_h
            q_mask_s = q_offs_s < seq_len
            qt_mask = q_mask_s[None, :] & qkv_mask_h[:, None]
            qt_tile = tl.load(Q_ptr + q_offs, mask=qt_mask).to(gemm_dtype) # [HEAD_DIM, BLOCK_SIZE_Q]
            m_offs = q_offs_s * m_stride_s
            m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(compute_dtype)[None, :] # [1, BLOCK_SIZE_Q]
            d_offs = q_offs_s * d_stride_s
            d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(compute_dtype)[None, :] # [1, BLOCK_SIZE_Q]
            dO_offs = (q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h)
            dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask_s[:, None] & qkv_mask_h[None, :]).to(compute_dtype) # [BLOCK_SIZE_Q, HEAD_DIM]
            if COMPUTE_DQ:
                dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)
            # Compute dv1.
            # [KV, D] @ [D, Q] => [KV, Q]
            qkkT = tl.dot(
                k1k2,
                qt_tile * softmax_scale,
                out_dtype=tl.float32,
            ) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

            # Mask qkkT to -inf.
            kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_offs_s[:, None]) & (kv1_offs_s[:, None] <= q_offs_s[None, :])
            kv2_local_mask = ((q_offs_s - w2) < kv2_idx) & (kv2_idx <= q_offs_s)
            local_mask = (kv1_local_mask & kv2_local_mask[None, :]) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            qkkT = tl.where(local_mask, qkkT, -1.0e38)

            pT = tl.exp(qkkT - m_tile) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            pT = tl.where(local_mask, pT, 0.0)
            dOv2 = dO_tile * v2_tile # [BLOCK_SIZE_Q, HEAD_DIM]
            dv1 += tl.dot(
                pT.to(gemm_dtype),
                dOv2.to(gemm_dtype),
                out_dtype=tl.float32,
            ) # [BLOCK_SIZE_KV, HEAD_DIM]

            dpT = tl.dot(
                v1v2,
                tl.trans(dO_tile.to(gemm_dtype)),
                out_dtype=tl.float32,
            ) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            dsT = pT * (dpT - d_tile) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            dsT = tl.where(local_mask, dsT, 0.0)
            dsT = dsT.to(gemm_dtype)

            dk1 += (
                tl.dot(dsT, tl.trans(qt_tile), out_dtype=tl.float32)
                * k2_tile.to(tl.float32)
                * softmax_scale
            )
            
            # Compute dQ: dQ = dsT^T @ (k1 * k2) * softmax_scale
            if COMPUTE_DQ:
                # dsT is [BLOCK_SIZE_KV, BLOCK_SIZE_Q], we need [BLOCK_SIZE_Q, BLOCK_SIZE_KV]
                # k1k2 is [BLOCK_SIZE_KV, HEAD_DIM] 
                # Result should be [BLOCK_SIZE_Q, HEAD_DIM]
                dq = tl.dot(
                    tl.trans(dsT),  # [BLOCK_SIZE_Q, BLOCK_SIZE_KV] 
                    k1k2.to(gemm_dtype),  # [BLOCK_SIZE_KV, HEAD_DIM]
                    out_dtype=tl.float32,
                ) * softmax_scale  # [BLOCK_SIZE_Q, HEAD_DIM]
                
                # Store dQ with atomic add since multiple KV pairs contribute to same Q positions
                dq_offs = (q_offs_s[:, None] * dq_stride_s + qkv_offs_h[None, :] * dq_stride_h)
                dq_mask = q_mask_s[:, None] & qkv_mask_h[None, :]
                # Use atomic add with float32 (atomic operations don't support bfloat16)
                tl.atomic_add(dQ_ptr + dq_offs, dq, mask=dq_mask)

    # Store accumulated dK1 and dV1 outside the loops (each kernel handles different kv1 positions)
    dv1_offs = kv1_offs_s[:, None] * dv1_stride_s + qkv_offs_h[None, :] * dv1_stride_h
    dk1_offs = kv1_offs_s[:, None] * dk1_stride_s + qkv_offs_h[None, :] * dk1_stride_h
    tl.store(dV1_ptr + dv1_offs, dv1.to(data_dtype), mask=kv1_mask)
    tl.store(dK1_ptr + dk1_offs, dk1.to(data_dtype), mask=kv1_mask)


# Backward pass for K2/V2 gradients  
@triton.autotune(configs=[Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 32}, num_warps=4)], key=["HEAD_DIM"])
@triton.jit
def two_simplicial_attn_bwd_kv2_kernel(
    Q_ptr, # [b, s, k, h]
    K1_ptr, # [b, s, k, h]
    K2_ptr, # [b, s, k, h]
    V1_ptr, # [b, s, k, h]
    V2_ptr, # [b, s, k, h]
    dO_ptr, # [b, s, k, h]
    M_ptr, # [b, k, s]
    D_ptr, # [b, k, s]
    dK2_ptr, # [b, s, k, h]
    dV2_ptr, # [b, s, k, h]
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1, # Q[i]: KV1(i-w1,i]
    w2, # Q[i]: KV2(i-w2,i]
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    dk2_stride_b,
    dk2_stride_s,
    dk2_stride_k,
    dk2_stride_h,
    dv2_stride_b,
    dv2_stride_s,
    dv2_stride_k,
    dv2_stride_h,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
):
    data_dtype = tl.bfloat16
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    kv2_start = tl.program_id(0) * BLOCK_SIZE_KV
    kv2_end = kv2_start + BLOCK_SIZE_KV
    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk
    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
    dK2_ptr += offs_b * dk2_stride_b + offs_k * dk2_stride_k
    dV2_ptr += offs_b * dv2_stride_b + offs_k * dv2_stride_k

    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    qkv_mask_h = qkv_offs_h < head_dim

    kv2_offs_s = kv2_start + tl.arange(0, BLOCK_SIZE_KV)

    k2_offs = kv2_offs_s[:, None] * k2_stride_s + qkv_offs_h[None, :] * k2_stride_h
    kv2_mask_s = kv2_offs_s < seq_len
    kv2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]
    k2_tile = tl.load(K2_ptr + k2_offs, mask=kv2_mask).to(compute_dtype) # [BLOCK_SIZE_KV, HEAD_DIM]
    v2_offs = kv2_offs_s[:, None] * v2_stride_s + qkv_offs_h[None, :] * v2_stride_h
    v2_tile = tl.load(V2_ptr + v2_offs, mask=kv2_mask).to(compute_dtype) # [BLOCK_SIZE_KV, HEAD_DIM]
    k2_tile += K2_BIAS
    v2_tile += V2_BIAS
    
    dv2 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)
    dk2 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)

    # Implement dK2/dV2 gradient computation (analogous to dK1/dV1 but for kv2 indices)
    # kv2 - w1 < kv1 <= kv2 + w2
    for kv1_idx in tl.range(tl.maximum(0, kv2_start - w1), tl.minimum(seq_len, kv2_end + w2)):
        k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
        k1_tile = (tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(compute_dtype))[None, :] # [1, HEAD_DIM]
        v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
        v1_tile = (tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(compute_dtype))[None, :] # [1, HEAD_DIM]
        
        k1k2 = k1_tile * k2_tile # [BLOCK_SIZE_KV, HEAD_DIM]
        v1v2 = v1_tile * v2_tile # [BLOCK_SIZE_KV, HEAD_DIM]
        k1k2 = k1k2.to(gemm_dtype)
        v1v2 = v1v2.to(gemm_dtype)

        # kv1 <= q < kv1 + w1
        # kv2 <= q < kv2 + w2  
        q_start = tl.maximum(kv1_idx, kv2_start)
        q_end = tl.minimum(seq_len, tl.minimum(kv1_idx + w1, kv2_end + w2))
        for q_idx in tl.range(q_start, q_end, BLOCK_SIZE_Q):
            # Load qt, m, d, dO
            q_offs_s = q_idx + tl.arange(0, BLOCK_SIZE_Q)
            q_offs = q_offs_s[None, :] * q_stride_s + qkv_offs_h[:, None] * q_stride_h
            q_mask_s = q_offs_s < seq_len
            qt_mask = q_mask_s[None, :] & qkv_mask_h[:, None]
            qt_tile = tl.load(Q_ptr + q_offs, mask=qt_mask).to(gemm_dtype) # [HEAD_DIM, BLOCK_SIZE_Q]
            m_offs = q_offs_s * m_stride_s
            m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(compute_dtype)[None, :] # [1, BLOCK_SIZE_Q]
            d_offs = q_offs_s * d_stride_s
            d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(compute_dtype)[None, :] # [1, BLOCK_SIZE_Q]
            dO_offs = (q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h)
            dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask_s[:, None] & qkv_mask_h[None, :]).to(compute_dtype) # [BLOCK_SIZE_Q, HEAD_DIM]
            
            # Compute attention scores and attention weights
            qkkT = tl.dot(
                k1k2,
                qt_tile * softmax_scale,
                out_dtype=tl.float32,
            ) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

            # Apply attention masks  
            kv1_local_mask = ((q_offs_s - w1) < kv1_idx) & (kv1_idx <= q_offs_s)  # [BLOCK_SIZE_Q]
            kv2_local_mask = ((q_offs_s[None, :] - w2) < kv2_offs_s[:, None]) & (kv2_offs_s[:, None] <= q_offs_s[None, :])  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            local_mask = kv1_local_mask[None, :] & kv2_local_mask  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            qkkT = tl.where(local_mask, qkkT, -1.0e38)

            pT = tl.exp(qkkT - m_tile) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            pT = tl.where(local_mask, pT, 0.0)
            
            # Compute dV2: dV2 += pT^T @ (dO * v1)
            dOv1 = dO_tile * v1_tile # [BLOCK_SIZE_Q, HEAD_DIM]
            dv2 += tl.dot(
                pT.to(gemm_dtype),
                dOv1.to(gemm_dtype),
                out_dtype=tl.float32,
            ) # [BLOCK_SIZE_KV, HEAD_DIM]

            # Compute dpT and dsT for dK2
            dpT = tl.dot(
                v1v2,
                tl.trans(dO_tile.to(gemm_dtype)),
                out_dtype=tl.float32,
            ) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            dsT = pT * (dpT - d_tile) # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            dsT = tl.where(local_mask, dsT, 0.0)
            dsT = dsT.to(gemm_dtype)

            # Compute dK2: dK2 += dsT @ qT * k1 * softmax_scale
            dk2 += (
                tl.dot(dsT, tl.trans(qt_tile), out_dtype=tl.float32)
                * k1_tile.to(tl.float32)
                * softmax_scale
            )

    # Store accumulated dK2 and dV2 gradients
    dv2_offs = kv2_offs_s[:, None] * dv2_stride_s + qkv_offs_h[None, :] * dv2_stride_h
    dk2_offs = kv2_offs_s[:, None] * dk2_stride_s + qkv_offs_h[None, :] * dk2_stride_h
    tl.store(dV2_ptr + dv2_offs, dv2.to(data_dtype), mask=kv2_mask)
    tl.store(dK2_ptr + dk2_offs, dk2.to(data_dtype), mask=kv2_mask)


def compute_D_kernel(dO, O):
    """Compute D tensor for backward pass: D[i] = sum_j(dO[i,j] * O[i,j])"""
    return torch.sum(dO * O, dim=-1) 