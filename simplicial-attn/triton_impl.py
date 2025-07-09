# Source: https://arxiv.org/pdf/2507.02754
# 2-SIMPLICIAL ATTENTION - TRITON IMPLEMENTATION WRAPPER

import torch
import triton
import math
from .kernels import (
    two_simplicial_attn_fwd_kernel,
    two_simplicial_attn_bwd_kv1_kernel,
    two_simplicial_attn_bwd_kv2_kernel,
    compute_D_kernel
)


class TwoSimplicialAttentionFunction(torch.autograd.Function):
    """PyTorch autograd Function for 2-simplicial attention with full gradient support."""
    
    @staticmethod
    def forward(ctx, q, k1, k2, v1, v2, w1, w2, scale, k2_bias, v2_bias):
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Initialize output and intermediate tensors
        output = torch.empty_like(q)
        M = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)
        D = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)
        
        # Configure grid
        BLOCK_SIZE_Q = 64
        BLOCK_SIZE_KV = 32
        grid = lambda META: (triton.cdiv(seq_len, BLOCK_SIZE_Q), batch_size * num_heads)
        
        # Launch forward kernel
        two_simplicial_attn_fwd_kernel[grid](
            q, k1, k2, v1, v2, output, M, D,
            batch_size, seq_len, num_heads, head_dim,
            w1, w2,
            # Strides
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  # q strides
            k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),  # k1 strides  
            k2.stride(0), k2.stride(1), k2.stride(2), k2.stride(3),  # k2 strides
            v1.stride(0), v1.stride(1), v1.stride(2), v1.stride(3),  # v1 strides
            v2.stride(0), v2.stride(1), v2.stride(2), v2.stride(3),  # v2 strides
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),  # output strides
            M.stride(0), M.stride(1), M.stride(2),  # M strides
            D.stride(0), D.stride(1), D.stride(2),  # D strides
            HEAD_DIM=head_dim,
            INPUT_PRECISION='tf32',
            SM_SCALE=scale,
            K2_BIAS=k2_bias,
            V2_BIAS=v2_bias,
        )
        
        # Save tensors for backward pass
        ctx.save_for_backward(q, k1, k2, v1, v2, output, M)
        ctx.w1, ctx.w2, ctx.scale, ctx.k2_bias, ctx.v2_bias = w1, w2, scale, k2_bias, v2_bias
        ctx.head_dim = head_dim
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k1, k2, v1, v2, output, M = ctx.saved_tensors
        w1, w2, scale, k2_bias, v2_bias = ctx.w1, ctx.w2, ctx.scale, ctx.k2_bias, ctx.v2_bias
        head_dim = ctx.head_dim
        
        batch_size, seq_len, num_heads, _ = q.shape
        
        # Compute D tensor needed for backward pass
        D = compute_D_kernel(grad_output, output)
        
        # Initialize gradient tensors
        # Use float32 for dQ to support atomic operations in Triton
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32) if ctx.needs_input_grad[0] else None
        dk1 = torch.zeros_like(k1) if ctx.needs_input_grad[1] else None
        dk2 = torch.zeros_like(k2) if ctx.needs_input_grad[2] else None
        dv1 = torch.zeros_like(v1) if ctx.needs_input_grad[3] else None
        dv2 = torch.zeros_like(v2) if ctx.needs_input_grad[4] else None
        
        # Configure grids for backward kernels
        BLOCK_SIZE_Q = 64
        BLOCK_SIZE_KV = 32
        grid_kv = lambda META: (triton.cdiv(seq_len, BLOCK_SIZE_KV), batch_size * num_heads)
        
        # Launch backward kernel for dQ, dK1, dV1
        if any([ctx.needs_input_grad[i] for i in [0, 1, 3]]):  # dq, dk1, dv1
            # Create dummy tensors for None gradients to avoid kernel errors
            dq_tensor = dq if dq is not None else torch.empty(q.shape, device=q.device, dtype=torch.float32)
            dk1_tensor = dk1 if dk1 is not None else torch.empty_like(k1)
            dv1_tensor = dv1 if dv1 is not None else torch.empty_like(v1)
            
            two_simplicial_attn_bwd_kv1_kernel[grid_kv](
                q, k1, k2, v1, v2, grad_output, M, D, dq_tensor, dk1_tensor, dv1_tensor,
                batch_size, seq_len, num_heads, head_dim,
                w1, w2,
                # All the strides...
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
                k2.stride(0), k2.stride(1), k2.stride(2), k2.stride(3),
                v1.stride(0), v1.stride(1), v1.stride(2), v1.stride(3),
                v2.stride(0), v2.stride(1), v2.stride(2), v2.stride(3),
                grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                M.stride(0), M.stride(1), M.stride(2),
                D.stride(0), D.stride(1), D.stride(2),
                dq_tensor.stride(0), dq_tensor.stride(1), dq_tensor.stride(2), dq_tensor.stride(3),
                dk1_tensor.stride(0), dk1_tensor.stride(1), dk1_tensor.stride(2), dk1_tensor.stride(3),
                dv1_tensor.stride(0), dv1_tensor.stride(1), dv1_tensor.stride(2), dv1_tensor.stride(3),
                HEAD_DIM=head_dim,
                SM_SCALE=scale,
                K2_BIAS=k2_bias,
                V2_BIAS=v2_bias,
                COMPUTE_DQ=ctx.needs_input_grad[0],
                is_flipped=False,
            )
            
            # Copy results back to actual gradient tensors
            if dq is not None:
                dq.copy_(dq_tensor)
            if dk1 is not None:
                dk1.copy_(dk1_tensor)
            if dv1 is not None:
                dv1.copy_(dv1_tensor)
        
        # Launch backward kernel for dK2, dV2
        if any([ctx.needs_input_grad[i] for i in [2, 4]]):  # dk2, dv2
            # Create dummy tensors for None gradients to avoid kernel errors
            dk2_tensor = dk2 if dk2 is not None else torch.empty_like(k2)
            dv2_tensor = dv2 if dv2 is not None else torch.empty_like(v2)
            
            two_simplicial_attn_bwd_kv2_kernel[grid_kv](
                q, k1, k2, v1, v2, grad_output, M, D, dk2_tensor, dv2_tensor,
                batch_size, seq_len, num_heads, head_dim,
                w1, w2,
                # All the strides...
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
                k2.stride(0), k2.stride(1), k2.stride(2), k2.stride(3),
                v1.stride(0), v1.stride(1), v1.stride(2), v1.stride(3),
                v2.stride(0), v2.stride(1), v2.stride(2), v2.stride(3),
                grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                M.stride(0), M.stride(1), M.stride(2),
                D.stride(0), D.stride(1), D.stride(2),
                dk2_tensor.stride(0), dk2_tensor.stride(1), dk2_tensor.stride(2), dk2_tensor.stride(3),
                dv2_tensor.stride(0), dv2_tensor.stride(1), dv2_tensor.stride(2), dv2_tensor.stride(3),
                HEAD_DIM=head_dim,
                SM_SCALE=scale,
                K2_BIAS=k2_bias,
                V2_BIAS=v2_bias,
            )
            
            # Copy results back to actual gradient tensors
            if dk2 is not None:
                dk2.copy_(dk2_tensor)
            if dv2 is not None:
                dv2.copy_(dv2_tensor)
        
        # Convert dQ back to original dtype and return gradients (None for non-tensor arguments)
        dq_output = dq.to(q.dtype) if dq is not None else None
        return dq_output, dk1, dk2, dv1, dv2, None, None, None, None, None


class TwoSimplicialAttention(torch.nn.Module):
    """PyTorch wrapper for Triton 2-simplicial attention kernels."""
    
    def __init__(self, head_dim, w1=64, w2=64, scale=None, k2_bias=0.0, v2_bias=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.w1 = w1
        self.w2 = w2
        self.scale = scale if scale is not None else (1.0 / math.sqrt(head_dim))
        self.k2_bias = k2_bias
        self.v2_bias = v2_bias
    
    def forward(self, q, k1, k2, v1, v2):
        """
        Forward pass using Triton kernels with full gradient support.
        
        Args:
            q, k1, k2, v1, v2: [batch, seq_len, num_heads, head_dim]
        
        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        return TwoSimplicialAttentionFunction.apply(
            q, k1, k2, v1, v2, 
            self.w1, self.w2, self.scale, self.k2_bias, self.v2_bias
        ) 