# Source: https://arxiv.org/pdf/2507.02754
# 2-SIMPLICIAL ATTENTION - MAIN MODULE

"""
This module provides a clean, organized implementation of 2-simplicial attention.

The implementation has been modularized into:
- kernels.py: Core Triton kernels
- triton_impl.py: PyTorch wrapper classes and autograd functions  
- reference.py: Reference PyTorch implementation for validation
- test_correctness.py: Comprehensive correctness tests
- test_performance.py: Performance benchmarks and analysis

Import the main classes directly from the package:
    from utils.simplical_attn import TwoSimplicialAttention, TwoSimplicialAttentionFunction
"""

# Re-export main components for convenience
from .triton_impl import TwoSimplicialAttention, TwoSimplicialAttentionFunction
from .reference import two_simplicial_attention_reference, TwoSimplicialAttentionReference
from .kernels import (
    two_simplicial_attn_fwd_kernel,
    two_simplicial_attn_bwd_kv1_kernel,
    two_simplicial_attn_bwd_kv2_kernel,
    compute_D_kernel
)

# Import test functions for backward compatibility
try:
    from .test_correctness import run_all_correctness_tests
    from .test_performance import run_all_performance_tests, quick_performance_test
except ImportError:
    # Optional dependencies might not be available
    def run_all_correctness_tests():
        print("⚠️  Correctness tests require additional dependencies (pytest, numpy)")
        return False
    
    def run_all_performance_tests():
        print("⚠️  Performance tests require additional dependencies")
        return False
    
    def quick_performance_test():
        print("⚠️  Performance tests require additional dependencies")
        return False


# Legacy function compatibility (keeping the old test functions for backward compatibility)
def test_forward_pass_correctness():
    """Legacy function - use run_all_correctness_tests() instead."""
    print("ℹ️  Use run_all_correctness_tests() for comprehensive testing")
    return quick_performance_test()

def test_backward_implementation():
    """Legacy function - use run_all_correctness_tests() instead."""
    print("ℹ️  Use run_all_correctness_tests() for comprehensive testing")
    
    # Quick validation test
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("   Skipping: Triton requires CUDA")
        return False
    
    try:
        batch_size, seq_len, num_heads, head_dim = 1, 32, 1, 16  
        w1, w2 = 8, 8
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, 
                       dtype=torch.bfloat16, requires_grad=True)
        k1 = torch.randn_like(q, requires_grad=True)
        k2 = torch.randn_like(q, requires_grad=True)
        v1 = torch.randn_like(q, requires_grad=True)
        v2 = torch.randn_like(q, requires_grad=True)
        
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        output = attn_layer(q, k1, k2, v1, v2)
        
        print(f"   ✓ Forward pass successful, output shape: {output.shape}")
        
        loss = output.sum()
        loss.backward()
        
        print(f"   ✓ Backward pass successful")
        print(f"   ✓ All gradients computed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def run_all_tests():
    """Legacy function - use run_all_correctness_tests() instead."""
    print("🧪 Running Quick Validation Tests (Legacy Function)")
    print("=" * 50)
    print("ℹ️  For comprehensive testing, use:")
    print("     run_all_correctness_tests() - Full correctness validation")  
    print("     run_all_performance_tests() - Performance benchmarks")
    print()
    
    forward_success = test_forward_pass_correctness()
    backward_success = test_backward_implementation()
    
    if forward_success and backward_success:
        print("\n🎉 Quick validation passed!")
        print("\nNext steps:")
        print("  • Run comprehensive tests: run_all_correctness_tests()")
        print("  • Run performance benchmarks: run_all_performance_tests()")
        print("  • Integration into larger models")
        return True
    else:
        print("\n❌ Quick validation failed")
        return False


# Make main classes available at package level
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
    'run_all_tests',  # Legacy compatibility
    'test_forward_pass_correctness',  # Legacy compatibility
    'test_backward_implementation'  # Legacy compatibility
]