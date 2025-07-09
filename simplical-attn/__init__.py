"""
2-Simplicial Attention Implementation

A high-performance Triton implementation of 2-simplicial attention with full backward pass support.

Main exports:
- TwoSimplicialAttention: Main PyTorch module
- TwoSimplicialAttentionFunction: Autograd function
- Triton kernels and reference implementations
"""

from .simplical_attn import (
    TwoSimplicialAttention, 
    TwoSimplicialAttentionFunction,
    two_simplicial_attention_reference,
    TwoSimplicialAttentionReference,
    two_simplicial_attn_fwd_kernel,
    two_simplicial_attn_bwd_kv1_kernel, 
    two_simplicial_attn_bwd_kv2_kernel,
    compute_D_kernel,
    run_all_correctness_tests,
    run_all_performance_tests,
    quick_performance_test,
    run_all_tests,
    test_forward_pass_correctness,
    test_backward_implementation
)

__all__ = [
    'TwoSimplicialAttention',
    'TwoSimplicialAttentionFunction', 
    'two_simplicial_attention_reference',
    'TwoSimplicialAttentionReference',
    'two_simplicial_attn_fwd_kernel',
    'two_simplicial_attn_bwd_kv1_kernel',
    'two_simplicial_attn_bwd_kv2_kernel',
    'compute_D_kernel',
    'run_all_correctness_tests',
    'run_all_performance_tests',
    'quick_performance_test',
    'run_all_tests',
    'test_forward_pass_correctness',
    'test_backward_implementation'
] 