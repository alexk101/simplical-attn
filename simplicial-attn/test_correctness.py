#!/usr/bin/env python3
"""
Comprehensive correctness tests for 2-simplicial attention.

Tests cover:
- All precision levels (float16, bfloat16, float32)
- Various input configurations and tensor shapes
- Edge cases and boundary conditions
- Gradient correctness via finite differences
- Numerical stability tests
"""

import torch
import pytest
from typing import Dict, List, Tuple

from .triton_impl import TwoSimplicialAttention
from .reference import two_simplicial_attention_reference


def get_device():
    """Get CUDA device if available, skip tests if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


class TestConfig:
    """Test configuration parameters."""
    
    # Test dimensions
    BATCH_SIZES = [1, 2, 4]
    SEQ_LENS = [16, 32, 64, 128, 256]
    NUM_HEADS = [1, 2, 4, 8]
    HEAD_DIMS = [16, 32, 64, 128, 256]
    WINDOW_SIZES = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
    
    # Precision levels
    DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    
    # Bias values to test
    BIAS_VALUES = [0.0, 0.1, -0.1, 1.0, -1.0]
    
    # Tolerances for different precisions
    TOLERANCES = {
        torch.float16: {'atol': 1e-2, 'rtol': 1e-2},
        torch.bfloat16: {'atol': 1e-2, 'rtol': 1e-2}, 
        torch.float32: {'atol': 1e-4, 'rtol': 1e-4}
    }


def create_test_inputs(batch_size: int, seq_len: int, num_heads: int, head_dim: int, 
                      dtype: torch.dtype, device: torch.device, 
                      input_magnitude: float = 0.1) -> Tuple[torch.Tensor, ...]:
    """Create test input tensors with controlled properties."""
    
    def create_tensor():
        return input_magnitude * torch.randn(
            batch_size, seq_len, num_heads, head_dim,
            device=device, dtype=dtype, requires_grad=True
        )
    
    return create_tensor(), create_tensor(), create_tensor(), create_tensor(), create_tensor()


def finite_difference_gradient_check(func, inputs: List[torch.Tensor], grad_output: torch.Tensor,
                                   eps: float = 1e-5, sample_size: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Compute finite difference gradients and compare with analytical gradients.
    
    Args:
        func: Function that computes attention output
        inputs: List of input tensors requiring gradients
        grad_output: Gradient w.r.t. output
        eps: Finite difference step size
        sample_size: Number of elements to sample for large tensors
        
    Returns:
        Dictionary with gradient comparison metrics for each input
    """
    results = {}
    
    # Get analytical gradients
    inputs_copy = [x.clone().detach().requires_grad_(True) for x in inputs]
    output = func(*inputs_copy)
    output.backward(grad_output)
    analytical_grads = [x.grad.clone() if x.grad is not None else None for x in inputs_copy]
    
    # Compute numerical gradients for each input
    for i, (inp, analytical_grad) in enumerate(zip(inputs, analytical_grads)):
        if analytical_grad is None:
            continue
            
        # Sample subset for efficiency on large tensors
        flat_inp = inp.view(-1)
        flat_analytical = analytical_grad.view(-1)
        
        if flat_inp.numel() > sample_size:
            indices = torch.randperm(flat_inp.numel(), device=inp.device)[:sample_size]
        else:
            indices = torch.arange(flat_inp.numel(), device=inp.device)
        
        numerical_grad = torch.zeros_like(flat_analytical)
        
        for idx in indices:
            # Compute f(x + eps)
            flat_inp[idx] += eps
            inputs_plus = inputs.copy()
            inputs_plus[i] = inp
            output_plus = func(*inputs_plus)
            loss_plus = torch.sum(output_plus * grad_output)
            
            # Compute f(x - eps)
            flat_inp[idx] -= 2 * eps
            inputs_minus = inputs.copy()
            inputs_minus[i] = inp
            output_minus = func(*inputs_minus)
            loss_minus = torch.sum(output_minus * grad_output)
            
            # Restore original value
            flat_inp[idx] += eps
            
            # Compute numerical derivative
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        # Compute error metrics
        sampled_numerical = numerical_grad[indices]
        sampled_analytical = flat_analytical[indices]
        
        abs_error = torch.abs(sampled_numerical - sampled_analytical)
        rel_error = abs_error / (torch.abs(sampled_analytical) + eps)
        
        results[f"input_{i}"] = {
            "max_abs_error": torch.max(abs_error).item(),
            "max_rel_error": torch.max(rel_error).item(),
            "mean_abs_error": torch.mean(abs_error).item(),
            "mean_rel_error": torch.mean(rel_error).item(),
            "analytical_norm": torch.norm(sampled_analytical).item(),
            "numerical_norm": torch.norm(sampled_numerical).item(),
        }
    
    return results


class TestForwardPassCorrectness:
    """Test forward pass correctness across configurations."""
    
    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", [
        (1, 32, 1, 16),   # Minimal case
        (2, 64, 2, 32),   # Small case
        (4, 128, 4, 64),  # Medium case
        (1, 256, 8, 128), # Large sequence
    ])
    @pytest.mark.parametrize("dtype", TestConfig.DTYPES)
    def test_forward_vs_reference(self, batch_size, seq_len, num_heads, head_dim, dtype):
        """Test Triton implementation matches reference for various configurations."""
        device = get_device()
        w1, w2 = 16, 16
        
        # Create inputs
        q, k1, k2, v1, v2 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        
        # Reference implementation (use float32 for precision)
        ref_output = two_simplicial_attention_reference(
            q.float(), k1.float(), k2.float(), v1.float(), v2.float(), w1, w2
        )
        
        # Triton implementation
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        triton_output = attn_layer(q, k1, k2, v1, v2)
        
        # Compare outputs
        tolerance = TestConfig.TOLERANCES[dtype]
        diff = torch.abs(ref_output.to(dtype) - triton_output).float()
        
        assert torch.allclose(ref_output.to(dtype), triton_output, **tolerance), \
            f"Outputs don't match for {dtype}: max_diff={torch.max(diff):.2e}, mean_diff={torch.mean(diff):.2e}"
    
    @pytest.mark.parametrize("w1,w2", TestConfig.WINDOW_SIZES)
    def test_window_sizes(self, w1, w2):
        """Test different window size combinations."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 64
        dtype = torch.bfloat16
        
        q, k1, k2, v1, v2 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        
        # Reference implementation
        ref_output = two_simplicial_attention_reference(
            q.float(), k1.float(), k2.float(), v1.float(), v2.float(), w1, w2
        )
        
        # Triton implementation
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        triton_output = attn_layer(q, k1, k2, v1, v2)
        
        tolerance = TestConfig.TOLERANCES[dtype]
        assert torch.allclose(ref_output.to(dtype), triton_output, **tolerance)
    
    @pytest.mark.parametrize("k2_bias,v2_bias", [
        (0.0, 0.0),
        (0.1, 0.2),
        (-0.1, -0.2),
        (1.0, 0.5),
    ])
    def test_bias_values(self, k2_bias, v2_bias):
        """Test different bias value combinations."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 2, 64, 2, 32
        w1, w2 = 16, 16
        dtype = torch.bfloat16
        
        q, k1, k2, v1, v2 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        
        # Reference implementation
        ref_output = two_simplicial_attention_reference(
            q.float(), k1.float(), k2.float(), v1.float(), v2.float(),
            w1, w2, k2_bias=k2_bias, v2_bias=v2_bias
        )
        
        # Triton implementation
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2, k2_bias=k2_bias, v2_bias=v2_bias)
        triton_output = attn_layer(q, k1, k2, v1, v2)
        
        tolerance = TestConfig.TOLERANCES[dtype]
        assert torch.allclose(ref_output.to(dtype), triton_output, **tolerance)


class TestBackwardPassCorrectness:
    """Test backward pass correctness and gradient computation."""
    
    def test_backward_pass_basic(self):
        """Test basic backward pass functionality."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 1, 32, 1, 16
        w1, w2 = 8, 8
        dtype = torch.bfloat16
        
        q, k1, k2, v1, v2 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        
        # Forward pass
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        output = attn_layer(q, k1, k2, v1, v2)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check all gradients exist and are not zero
        for name, tensor in [("q", q), ("k1", k1), ("k2", k2), ("v1", v1), ("v2", v2)]:
            assert tensor.grad is not None, f"{name} gradient is None"
            assert not torch.allclose(tensor.grad, torch.zeros_like(tensor.grad)), f"{name} gradient is zero"
            assert not torch.isnan(tensor.grad).any(), f"{name} gradient contains NaN"
            assert not torch.isinf(tensor.grad).any(), f"{name} gradient contains Inf"
    
    @pytest.mark.parametrize("dtype", [torch.float32])  # Use float32 for better finite diff accuracy
    def test_gradient_correctness_finite_diff(self, dtype):
        """Test gradient correctness using finite differences."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 1, 16, 1, 16
        w1, w2 = 4, 4
        
        # Create small magnitude inputs for numerical stability
        q, k1, k2, v1, v2 = create_test_inputs(
            batch_size, seq_len, num_heads, head_dim, dtype, device, input_magnitude=0.01
        )
        inputs = [q, k1, k2, v1, v2]
        
        # Test function
        def attention_func(q, k1, k2, v1, v2):
            attn = TwoSimplicialAttention(head_dim, w1, w2)
            return attn(q, k1, k2, v1, v2)
        
        # Forward pass and create grad output
        output = attention_func(*inputs)
        grad_output = 0.01 * torch.randn_like(output)
        
        # Run finite difference check
        results = finite_difference_gradient_check(attention_func, inputs, grad_output, eps=1e-4)
        
        # Validate results
        for input_name, metrics in results.items():
            # Allow relaxed tolerances for complex attention operations
            assert metrics["max_rel_error"] < 0.2, f"{input_name}: max_rel_error={metrics['max_rel_error']:.2e}"
            assert metrics["mean_rel_error"] < 0.1, f"{input_name}: mean_rel_error={metrics['mean_rel_error']:.2e}"
            
            print(f"{input_name}: max_rel_err={metrics['max_rel_error']:.2e}, "
                  f"mean_rel_err={metrics['mean_rel_error']:.2e}")
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly across multiple backward passes."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 1, 16, 1, 16
        w1, w2 = 4, 4
        dtype = torch.bfloat16
        
        q, k1, k2, v1, v2 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        
        # First backward pass
        output1 = attn_layer(q, k1, k2, v1, v2)
        loss1 = output1.sum()
        loss1.backward(retain_graph=True)
        
        # Store first gradients
        first_grads = [tensor.grad.clone() for tensor in [q, k1, k2, v1, v2]]
        
        # Second backward pass (should accumulate)
        output2 = attn_layer(q, k1, k2, v1, v2)
        loss2 = output2.sum()
        loss2.backward()
        
        # Check gradients accumulated
        for i, tensor in enumerate([q, k1, k2, v1, v2]):
            expected_grad = 2 * first_grads[i]  # Should be doubled
            assert torch.allclose(tensor.grad, expected_grad, atol=1e-4), \
                f"Gradient accumulation failed for tensor {i}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimal_dimensions(self):
        """Test with minimal tensor dimensions."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 1, 16, 1, 16  # Minimum for Triton
        w1, w2 = 1, 1  # Minimal windows
        dtype = torch.bfloat16
        
        q, k1, k2, v1, v2 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        output = attn_layer(q, k1, k2, v1, v2)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_large_windows(self):
        """Test with window sizes larger than sequence length."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 1, 32, 1, 16
        w1, w2 = 100, 100  # Larger than seq_len
        dtype = torch.bfloat16
        
        q, k1, k2, v1, v2 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        output = attn_layer(q, k1, k2, v1, v2)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_zero_inputs(self):
        """Test with zero input tensors."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 1, 16, 1, 16
        w1, w2 = 4, 4
        dtype = torch.bfloat16
        
        q = torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        k1 = torch.zeros_like(q, requires_grad=True)
        k2 = torch.zeros_like(q, requires_grad=True)
        v1 = torch.zeros_like(q, requires_grad=True)
        v2 = torch.zeros_like(q, requires_grad=True)
        
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        output = attn_layer(q, k1, k2, v1, v2)
        
        # Should handle gracefully
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_extreme_values(self):
        """Test with extreme input values."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 1, 16, 1, 16
        w1, w2 = 4, 4
        dtype = torch.bfloat16
        
        # Large values
        large_val = 10.0
        q = large_val * torch.ones(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        k1 = large_val * torch.ones_like(q, requires_grad=True)
        k2 = large_val * torch.ones_like(q, requires_grad=True)
        v1 = large_val * torch.ones_like(q, requires_grad=True)
        v2 = large_val * torch.ones_like(q, requires_grad=True)
        
        attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
        output = attn_layer(q, k1, k2, v1, v2)
        
        # Should not overflow to inf/nan
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestNumericalStability:
    """Test numerical stability properties."""
    
    def test_deterministic_output(self):
        """Test that outputs are deterministic given fixed seeds."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 2, 32, 2, 32
        w1, w2 = 8, 8
        dtype = torch.bfloat16
        
        # First run
        torch.manual_seed(42)
        q1, k11, k21, v11, v21 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        attn_layer1 = TwoSimplicialAttention(head_dim, w1, w2)
        output1 = attn_layer1(q1, k11, k21, v11, v21)
        
        # Second run with same seed
        torch.manual_seed(42)
        q2, k12, k22, v12, v22 = create_test_inputs(batch_size, seq_len, num_heads, head_dim, dtype, device)
        attn_layer2 = TwoSimplicialAttention(head_dim, w1, w2)
        output2 = attn_layer2(q2, k12, k22, v12, v22)
        
        assert torch.allclose(output1, output2), "Outputs not deterministic"
    
    def test_precision_consistency(self):
        """Test that different precisions give consistent results within tolerance."""
        device = get_device()
        batch_size, seq_len, num_heads, head_dim = 1, 32, 1, 16
        w1, w2 = 8, 8
        
        # Create inputs in float32
        torch.manual_seed(42)
        q_f32, k1_f32, k2_f32, v1_f32, v2_f32 = create_test_inputs(
            batch_size, seq_len, num_heads, head_dim, torch.float32, device
        )
        
        # Test different precisions
        results = {}
        for dtype in [torch.float32, torch.bfloat16]:
            q = q_f32.to(dtype).detach().requires_grad_(True)
            k1 = k1_f32.to(dtype).detach().requires_grad_(True)
            k2 = k2_f32.to(dtype).detach().requires_grad_(True)
            v1 = v1_f32.to(dtype).detach().requires_grad_(True)
            v2 = v2_f32.to(dtype).detach().requires_grad_(True)
            
            attn_layer = TwoSimplicialAttention(head_dim, w1, w2)
            output = attn_layer(q, k1, k2, v1, v2)
            results[dtype] = output.float()
        
        # Compare float32 vs bfloat16
        diff = torch.abs(results[torch.float32] - results[torch.bfloat16])
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        # bfloat16 should be reasonably close to float32
        assert max_diff < 0.1, f"Precision inconsistency: max_diff={max_diff:.2e}"
        assert mean_diff < 0.01, f"Precision inconsistency: mean_diff={mean_diff:.2e}"


def run_all_correctness_tests():
    """Run all correctness tests."""
    print("🧪 Running Comprehensive 2-Simplicial Attention Correctness Tests")
    print("=" * 70)
    
    # Run pytest with specific test classes
    test_classes = [
        TestForwardPassCorrectness,
        TestBackwardPassCorrectness,
        TestEdgeCases,
        TestNumericalStability
    ]
    
    failed_tests = []
    passed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        try:
            # Instantiate test class and run methods
            test_instance = test_class()
            
            for method_name in dir(test_instance):
                if method_name.startswith('test_'):
                    print(f"  🔍 {method_name}...")
                    try:
                        method = getattr(test_instance, method_name)
                        if hasattr(method, 'pytestmark'):
                            # Skip parametrized tests for now (would need pytest runner)
                            print(f"    ⏭️  Skipping parametrized test")
                            continue
                        method()
                        print(f"    ✅ PASSED")
                        passed_tests.append(f"{test_class.__name__}.{method_name}")
                    except Exception as e:
                        print(f"    ❌ FAILED: {e}")
                        failed_tests.append(f"{test_class.__name__}.{method_name}: {e}")
                        
        except Exception as e:
            print(f"  ❌ Test class failed: {e}")
            failed_tests.append(f"{test_class.__name__}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Passed: {len(passed_tests)}")
    print(f"❌ Failed: {len(failed_tests)}")
    
    if passed_tests:
        print(f"\n✅ PASSED TESTS:")
        for test in passed_tests:
            print(f"  • {test}")
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for test in failed_tests:
            print(f"  • {test}")
        return False
    else:
        print(f"\n🎉 ALL TESTS PASSED!")
        return True


if __name__ == "__main__":
    success = run_all_correctness_tests()
    exit(0 if success else 1) 