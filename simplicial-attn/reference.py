# Source: https://arxiv.org/pdf/2507.02754
# 2-SIMPLICIAL ATTENTION - PYTORCH REFERENCE IMPLEMENTATION

import torch
import math


def two_simplicial_attention_reference(q, k1, k2, v1, v2, w1, w2, scale=None, k2_bias=0.0, v2_bias=0.0):
    """
    Reference implementation of 2-simplicial attention in PyTorch for testing.
    
    This is the gold standard implementation used to validate the Triton kernels.
    It's intentionally written for clarity and correctness, not performance.
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k1, k2: Key tensors [batch, seq_len, num_heads, head_dim]
        v1, v2: Value tensors [batch, seq_len, num_heads, head_dim]
        w1, w2: Window sizes for local attention
        scale: Attention scale factor
        k2_bias, v2_bias: Bias terms for k2 and v2
    
    Returns:
        output: Attention output [batch, seq_len, num_heads, head_dim]
        attention_weights: For debugging [batch, num_heads, seq_len, seq_len, seq_len]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Add bias to k2 and v2
    k2 = k2 + k2_bias
    v2 = v2 + v2_bias
    
    output = torch.zeros_like(q)
    
    # For each position i in the sequence
    for i in range(seq_len):
        # Define windows for k1 and k2
        k1_start = max(0, i - w1)
        k1_end = min(seq_len, i + 1)  # inclusive of i
        
        k2_start = max(0, i - w2) 
        k2_end = min(seq_len, i + 1)  # inclusive of i
        
        # Get query for position i
        qi = q[:, i:i+1, :, :]  # [batch, 1, num_heads, head_dim]
        
        attention_sum = torch.zeros_like(qi)
        total_weight = torch.zeros(batch_size, 1, num_heads, 1, device=q.device)
        
        # Iterate over k1 window
        for j1 in range(k1_start, k1_end):
            k1j = k1[:, j1:j1+1, :, :]  # [batch, 1, num_heads, head_dim]
            v1j = v1[:, j1:j1+1, :, :]  # [batch, 1, num_heads, head_dim]
            
            # Iterate over k2 window  
            for j2 in range(k2_start, k2_end):
                k2j = k2[:, j2:j2+1, :, :]  # [batch, 1, num_heads, head_dim]
                v2j = v2[:, j2:j2+1, :, :]  # [batch, 1, num_heads, head_dim]
                
                # Compute attention score: q * k1 * k2 (element-wise)
                score = torch.sum(qi * k1j * k2j * scale, dim=-1, keepdim=True)  # [batch, 1, num_heads, 1]
                weight = torch.exp(score)
                
                # Compute value: v1 * v2 (element-wise)
                value = v1j * v2j  # [batch, 1, num_heads, head_dim]
                
                attention_sum += weight * value
                total_weight += weight
        
        # Normalize
        output[:, i:i+1, :, :] = attention_sum / (total_weight + 1e-8)
    
    return output


class TwoSimplicialAttentionReference(torch.nn.Module):
    """PyTorch module wrapper for the reference implementation."""
    
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
        Forward pass using reference PyTorch implementation.
        
        Args:
            q, k1, k2, v1, v2: [batch, seq_len, num_heads, head_dim]
        
        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        return two_simplicial_attention_reference(
            q, k1, k2, v1, v2, 
            self.w1, self.w2, self.scale, self.k2_bias, self.v2_bias
        ) 